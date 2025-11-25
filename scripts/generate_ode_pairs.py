from utils.distributed import launch_distributed_job
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from utils.dataset import TextDataset
import torch.distributed as dist
from tqdm import tqdm
import argparse
import torch
import math
import os


def init_model(device, checkpoint_path=None):
    model = WanDiffusionWrapper().to(device).to(torch.float32)
    encoder = WanTextEncoder().to(device).to(torch.float32)
    model.model.requires_grad_(False)
    encoder.requires_grad_(False)

    if checkpoint_path:
        print(f"Loading Self-Forcing checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # 处理可能的 key 不匹配问题
        if "generator" in state_dict:
            weights = state_dict["generator"]
        elif "model" in state_dict:
            weights = state_dict["model"]
        elif "generator_ema" in state_dict:
            weights = state_dict["generator_ema"]
        else:
            weights = state_dict

        missing, unexpected = model.load_state_dict(weights, strict=False)
        print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print("WARNING: Using default initialized weights! Make sure this is what you want.")

    scheduler = FlowMatchScheduler(
        shift=8.0, sigma_min=0.0, extra_one_step=True)
    
    # [关键设定] 保持4步推理，提高速度，对齐Teacher能力
    scheduler.set_timesteps(num_inference_steps=4, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)

    sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

    unconditional_dict = encoder(
        text_prompts=[sample_neg_prompt]
    )

    return model, encoder, scheduler, unconditional_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--caption_path", type=str)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained Self-Forcing model")
    parser.add_argument("--guidance_scale", type=float, default=6.0)

    args = parser.parse_args()

    # launch_distributed_job()
    launch_distributed_job()

    device = torch.cuda.current_device()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, encoder, scheduler, unconditional_dict = init_model(device=device, checkpoint_path=args.checkpoint_path)

    dataset = TextDataset(args.caption_path)

    # if global_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

    # ================= [新增配置] =================
    # 设置您想要生成的总条数
    TOTAL_LIMIT = 8000 
    
    # 计算每张卡需要跑多少条 (比如 5000条 / 2卡 = 2500条/卡)
    limit_per_gpu = TOTAL_LIMIT // dist.get_world_size()
    # ============================================

    for index in tqdm(range(int(math.ceil(len(dataset) / dist.get_world_size()))), disable=dist.get_rank() != 0):
        
        # ================= [新增自动停止逻辑] =================
        # 如果当前显卡处理的索引超过了分配的任务量，就停止
        if index >= limit_per_gpu:
            if dist.get_rank() == 0:
                print(f"\n[INFO] 已达到设定数量 ({TOTAL_LIMIT} 条)，停止生成。")
            break
        # ====================================================

        prompt_index = index * dist.get_world_size() + dist.get_rank()
        if prompt_index >= len(dataset):
            continue
        prompt = dataset[prompt_index]

        # 简单的去重/文件名处理
        safe_prompt_name = f"{prompt_index:05d}"
        save_path = os.path.join(args.output_folder, f"{safe_prompt_name}.pt")
        if os.path.exists(save_path):
            continue
        
        conditional_dict = encoder(text_prompts=prompt)

        latents = torch.randn(
            [1, 21, 16, 60, 104], dtype=torch.float32, device=device
        )

        noisy_input = []

        for progress_id, t in enumerate(scheduler.timesteps):
            timestep = t * \
                torch.ones([1, 21], device=device, dtype=torch.float32)

            noisy_input.append(latents)

            _, x0_pred_cond = model(
                latents, conditional_dict, timestep
            )

            _, x0_pred_uncond = model(
                latents, unconditional_dict, timestep
            )

            x0_pred = x0_pred_uncond + args.guidance_scale * (
                x0_pred_cond - x0_pred_uncond
            )

            flow_pred = model._convert_x0_to_flow_pred(
                scheduler=scheduler,
                x0_pred=x0_pred.flatten(0, 1),
                xt=latents.flatten(0, 1),
                timestep=timestep.flatten(0, 1)
            ).unflatten(0, x0_pred.shape[:2])

            latents = scheduler.step(
                flow_pred.flatten(0, 1),
                scheduler.timesteps[progress_id] * torch.ones(
                    [1, 21], device=device, dtype=torch.long).flatten(0, 1),
                latents.flatten(0, 1)
            ).unflatten(dim=0, sizes=flow_pred.shape[:2])

        noisy_input.append(latents)

        noisy_inputs = torch.stack(noisy_input, dim=1)

        # 4步推理不需要切片，保留所有5个节点
        noisy_inputs = noisy_inputs

        stored_data = noisy_inputs

        torch.save(
            {"ode_latent": stored_data, "prompts": prompt},
            save_path
        )

    dist.barrier()


if __name__ == "__main__":
    main()