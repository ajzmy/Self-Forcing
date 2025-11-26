import gc
import logging
# from peft import get_peft_model, LoraConfig, TaskType

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD
import torch
import wandb
import time
import os
from torch.distributed.fsdp import CPUOffload
import shutil


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # Calculate Gradient Accumulation Steps
        # 自动计算梯度累积步数
        self.micro_batch_size = config.batch_size * self.world_size
        self.grad_accum_steps = max(1, config.total_batch_size // self.micro_batch_size)
        
        if self.is_main_process:
            print(f"=== Gradient Accumulation Configuration ===")
            print(f"Total Batch Size Target: {config.total_batch_size}")
            print(f"Physical GPUs: {self.world_size}")
            print(f"Per-GPU Batch Size: {config.batch_size}")
            print(f"Effective Micro Batch Size: {self.micro_batch_size}")
            print(f"Calculated Accumulation Steps: {self.grad_accum_steps}")
            print(f"=========================================")

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # ======================= 全量微调设置 (移除 LoRA) =======================
        print("Configuring for Full Fine-tuning...")
        
        # Generator: 开启梯度
        self.model.generator.requires_grad_(True)
        # Teacher (Real Score): 冻结
        self.model.real_score.requires_grad_(False)
        # Critic (Fake Score): 开启梯度
        self.model.fake_score.requires_grad_(True)
        
        if self.is_main_process:
            gen_params = sum(p.numel() for p in self.model.generator.parameters() if p.requires_grad)
            critic_params = sum(p.numel() for p in self.model.fake_score.parameters() if p.requires_grad)
            print(f"Generator Trainable Params: {gen_params / 1e9:.2f} B")
            print(f"Critic Trainable Params: {critic_params / 1e9:.2f} B")

        # ======================= FSDP 包装 =======================
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=CPUOffload(offload_params=True)
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=CPUOffload(offload_params=True)
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=CPUOffload(offload_params=True)
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", True)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # ======================= 检查点加载 =======================
        if getattr(config, "resume_step", 0) > 0:
            self.step = config.resume_step
            print(f"Resuming training from step {self.step}")
            ckpt_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            ckpt_path = os.path.join(ckpt_dir, "model.pt")
            if os.path.exists(ckpt_path):
                print(f"Loading full checkpoint state from {ckpt_path}")
                state_dict = torch.load(ckpt_path, map_location="cpu")
                
                if "generator" in state_dict:
                    self.model.generator.load_state_dict(state_dict["generator"], strict=False)
                if "critic" in state_dict:
                    self.model.fake_score.load_state_dict(state_dict["critic"], strict=False)
            else:
                print(f"Warning: Resume step is set but checkpoint {ckpt_path} not found.")

        # ======================= 初始化优化器 =======================
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        # 6. Set up EMA
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue
            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
            
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(self.model.generator)
        critic_state_dict = fsdp_state_dict(self.model.fake_score)

        state_dict = {
            "generator": generator_state_dict,
            "critic": critic_state_dict,
        }
        
        if self.generator_ema is not None:
             state_dict["generator_ema"] = self.generator_ema.state_dict()

        if self.is_main_process:
            current_checkpoint_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(current_checkpoint_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(current_checkpoint_dir, "model.pt"))
            print("Model saved to", os.path.join(current_checkpoint_dir, "model.pt"))
            
            # 清理旧检查点
            all_checkpoints = [d for d in os.listdir(self.output_path) if d.startswith('checkpoint_model_')]
            all_checkpoints.sort(key=lambda x: int(x.split('_')[-1]))
            
            checkpoints_to_keep = 10 
            if len(all_checkpoints) > checkpoints_to_keep:
                checkpoints_to_delete = all_checkpoints[:-checkpoints_to_keep]
                for ckpt_dir in checkpoints_to_delete:
                    full_path = os.path.join(self.output_path, ckpt_dir)
                    print(f"Removing old checkpoint: {full_path}")
                    shutil.rmtree(full_path)

    def fwdbwd_one_step(self, batch, train_generator, accum_steps=1):
        """
        Modified forward/backward step to support gradient accumulation.
        NOTE: This function now performs LOSS SCALING but does NOT clip gradients here.
        """
        self.model.eval()  # prevent any randomness (e.g. dropout)

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Train Generator
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )
            
            # Scale Loss for Accumulation
            scaled_loss = generator_loss / accum_steps
            scaled_loss.backward()

            # We return the RAW loss for logging, but the gradients are already scaled and backwarded
            generator_log_dict.update({"generator_loss": generator_loss})
            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Train Critic
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        # Scale Loss for Accumulation
        scaled_critic_loss = critic_loss / accum_steps
        scaled_critic_loss.backward()

        critic_log_dict.update({"critic_loss": critic_loss})
        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        start_step = self.step

        while self.step < self.config.max_steps:
            if self.step % 20 == 0:
                torch.cuda.empty_cache()

            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # =================== Train Generator ===================
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                
                # Gradient Accumulation Loop
                for _ in range(self.grad_accum_steps):
                    batch = next(self.dataloader)
                    extra = self.fwdbwd_one_step(batch, True, accum_steps=self.grad_accum_steps)
                    extras_list.append(extra)
                
                # Clip Gradients AFTER accumulation
                generator_grad_norm = self.model.generator.clip_grad_norm_(
                    self.max_grad_norm_generator)
                
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
                    
                generator_log_dict = merge_dict_list(extras_list)
                # Manually add grad norm to log dict for logging consistency
                generator_log_dict["generator_grad_norm"] = generator_grad_norm

            # =================== Train Critic ===================
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            
            # Gradient Accumulation Loop
            for _ in range(self.grad_accum_steps):
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, False, accum_steps=self.grad_accum_steps)
                extras_list.append(extra)

            # Clip Gradients AFTER accumulation
            critic_grad_norm = self.model.fake_score.clip_grad_norm_(
                self.max_grad_norm_critic)
            
            self.critic_optimizer.step()
            
            critic_log_dict = merge_dict_list(extras_list)
            critic_log_dict["critic_grad_norm"] = critic_grad_norm

            # =================== Logging & Housekeeping ===================
            self.step += 1

            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].item(), # it's already a scalar from clip_grad_norm_
                            "dmdtrain_gradient_norm": generator_log_dict.get("dmdtrain_gradient_norm", torch.tensor(0.0)).mean().item()
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)
                else:
                    log_str = f"Step: {self.step}"
                    for key, value in wandb_loss_dict.items():
                        log_str += f", {key}: {value:.6f}"
                    print(log_str)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time