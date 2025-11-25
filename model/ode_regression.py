import torch.nn.functional as F
from typing import Tuple
import torch

from model.base import BaseModel
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class ODERegression(BaseModel):
    def __init__(self, args, device):
        """
        Initialize the ODERegression module.
        This class is self-contained and compute generator losses
        in the forward pass given precomputed ode solution pairs.
        This class supports the ode regression loss for both causal and bidirectional models.
        See Sec 4.3 of CausVid https://arxiv.org/abs/2412.07772 for details
        """
        super().__init__(args, device)

        # Step 1: Initialize all models

        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)
        if getattr(args, "generator_ckpt", False):
            print(f"Loading pretrained generator from {args.generator_ckpt}")
            state_dict = torch.load(args.generator_ckpt, map_location="cpu")[
                'generator']
            self.generator.load_state_dict(
                state_dict, strict=True
            )

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)

    """ def _initialize_models(self, args):
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False) """

    @torch.no_grad()
    def _prepare_generator_input(self, ode_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size].
        """
        batch_size, num_denoising_steps, num_frames, num_channels, height, width = ode_latent.shape

        # Step 1: Randomly choose a timestep for each frame
        index = self._get_timestep(
            0,
            len(self.denoising_step_list),
            batch_size,
            num_frames,
            self.num_frame_per_block,
            uniform_timestep=False
        )
        if self.args.i2v:
            index[:, 0] = len(self.denoising_step_list) - 1

        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(batch_size, 1, num_frames, 1, 1, 1).expand(
                -1, -1, -1, num_channels, height, width).to(self.device)
        ).squeeze(1)

        timestep = self.denoising_step_list[index].to(self.device)

        # if self.extra_noise_step > 0:
        #     random_timestep = torch.randint(0, self.extra_noise_step, [
        #                                     batch_size, num_frames], device=self.device, dtype=torch.long)
        #     perturbed_noisy_input = self.scheduler.add_noise(
        #         noisy_input.flatten(0, 1),
        #         torch.randn_like(noisy_input.flatten(0, 1)),
        #         random_timestep.flatten(0, 1)
        #     ).detach().unflatten(0, (batch_size, num_frames)).type_as(noisy_input)

        #     noisy_input[timestep == 0] = perturbed_noisy_input[timestep == 0]

        return noisy_input, timestep

    def generator_loss(self, ode_latent: torch.Tensor, conditional_dict: dict) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noisy latents and compute the ODE regression loss.
        Input:
            - ode_latent: a tensor containing the ODE latents [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
            They are ordered from most noisy to clean latents.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - log_dict: a dictionary containing additional information for loss timestep breakdown.
        """
        # Step 1: Run generator on noisy latents
        target_latent = ode_latent[:, -1]

        noisy_input, timestep = self._prepare_generator_input(
            ode_latent=ode_latent)

        _, pred_image_or_video = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        # Step 2: Compute the regression loss
        mask = timestep != 0

        loss = F.mse_loss(
            pred_image_or_video[mask], target_latent[mask], reduction="mean")

        log_dict = {
            "unnormalized_loss": F.mse_loss(pred_image_or_video, target_latent, reduction='none').mean(dim=[1, 2, 3, 4]).detach(),
            "timestep": timestep.float().mean(dim=1).detach(),
            "input": noisy_input.detach(),
            "output": pred_image_or_video.detach(),
        }

        return loss, log_dict


    def generator_loss_mixed_steps(self, ode_latent: torch.Tensor, conditional_dict: dict) -> Tuple[torch.Tensor, dict]:
        """
        混合步数训练 Loss (修正版):
        Block 0: 从 4 步中随机选一步作为输入 (t=1000, 750, 500, 250)，预测 GT。
        Block >0: 强制从纯噪声(t=1000) 作为输入，预测 GT。
        
        关键修正：所有情况下的 Target 都是 ode_latent[:, -1] (Clean Data / Ground Truth)。
        """
        batch_size, num_steps, num_frames, num_channels, height, width = ode_latent.shape
        num_frame_per_block = self.num_frame_per_block # 通常是 3

        # 1. 区分 Block 0 和 其他 Block
        block_indices = torch.arange(num_frames, device=self.device) // num_frame_per_block
        is_first_block = (block_indices == 0)

        # 2. 构造输入时间步 (Input Indices)
        # -----------------------------------------------------------
        # Block 0: 随机选中间状态 [0, num_steps-2]
        # (即从 Noisy 到 倒数第二步 之间选一个作为输入)
        t_indices_first = torch.randint(0, num_steps - 1, (batch_size,), device=self.device)
        
        # Block >0: 强制选初始噪声 (Index 0, T=1000)
        t_indices_rest = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # 组合 indices
        t_indices = torch.zeros((batch_size, num_frames), dtype=torch.long, device=self.device)
        for b in range(batch_size):
            t_indices[b, is_first_block] = t_indices_first[b]
            t_indices[b, ~is_first_block] = t_indices_rest[b]

        # 3. 构造 Target (修正：始终是 Ground Truth)
        # -----------------------------------------------------------
        # 无论是第一块还是后面块，无论是哪一步，目标都是预测干净的视频
        # ode_latent 的最后一个维度 (-1) 存放的是 Ground Truth (T=0)
        target_latent = ode_latent[:, -1] 

        # 4. 根据 Index 取出 Input Tensor
        # -----------------------------------------------------------
        gather_shape = (batch_size, 1, num_frames, 1, 1, 1)
        expand_shape = (-1, -1, -1, num_channels, height, width)

        # 取 Input (Noisy Latent)
        gather_idx_in = t_indices.view(*gather_shape).expand(*expand_shape)
        noisy_input = torch.gather(ode_latent, 1, gather_idx_in).squeeze(1)

        # 获取对应的 Timestep 值 (用于 Conditioning)
        timestep_val = self.denoising_step_list[t_indices].to(self.device)

        # 5. 前向计算 Loss
        # -----------------------------------------------------------
        # 模型输入噪声，输出预测的 Clean Latent (pred_output)
        _, pred_output = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep_val
        )

        # 计算预测值与真实值 (GT) 的 MSE Loss
        loss = F.mse_loss(pred_output, target_latent, reduction="mean")
        
        return loss, {"loss": loss.detach()}

    