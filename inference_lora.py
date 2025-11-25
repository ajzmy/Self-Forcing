import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from peft import get_peft_model, LoraConfig # 新增依赖

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

# --- 1. 参数定义部分 ---
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the base checkpoint (optional if included in config)")
# 新增: LoRA 权重路径
parser.add_argument("--lora_checkpoint_path", type=str, required=True, help="Path to the LoRA checkpoint (folder or pt file)") 
parser.add_argument("--data_path", type=str, help="Path to the dataset (prompt file)")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--visualization_path", type=str, default=None, help="Folder to save intermediate denoising steps.")
parser.add_argument("--num_output_frames", type=int, default=21, help="Number of overlap frames")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters") # LoRA通常不用EMA，但保留参数
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per prompt")
parser.add_argument("--save_with_index", action="store_true", help="Save with index")
args = parser.parse_args()

# --- 2. 初始化环境 ---
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40
torch.set_grad_enabled(False)

# --- 3. 加载配置 ---
config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# --- 4. 初始化 Pipeline (加载 Base 模型) ---
if hasattr(config, 'denoising_step_list'):
    pipeline = CausalInferencePipeline(config, device=device, visualization_path=args.visualization_path)
else:
    pipeline = CausalDiffusionInferencePipeline(config, device=device, visualization_path=args.visualization_path)

# 如果有 Base Model 的 checkpoint，先加载它 (通常 config 里有默认模型，这里是覆盖)
if args.checkpoint_path:
    print(f"Loading base model from {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])

# --- 5. 核心修改：加载 LoRA 权重 ---
print(f"Applying LoRA and loading weights from {args.lora_checkpoint_path}")

# (A) 定义 LoRA 配置 (必须与训练代码 distillation.py 中的配置一致)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
)

# (B) 将 LoRA 结构应用到 Generator
# 注意：pipeline.generator 是 WanDiffusionWrapper，训练时也是对它加的 LoRA
pipeline.generator = get_peft_model(pipeline.generator, lora_config)

# (C) 加载 LoRA 权重文件
lora_path = args.lora_checkpoint_path
if os.path.isdir(lora_path):
    lora_path = os.path.join(lora_path, "model.pt")

if os.path.exists(lora_path):
    checkpoint = torch.load(lora_path, map_location="cpu")
    
    # 从字典中提取 generator 部分
    if "generator" in checkpoint:
        lora_state_dict = checkpoint["generator"]
    else:
        lora_state_dict = checkpoint

    # 处理可能的 FSDP 包装产生的 key 前缀问题 (如 _fsdp_wrapped_module)
    new_state_dict = {}
    for k, v in lora_state_dict.items():
        new_k = k.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
        new_state_dict[new_k] = v
        
    # 加载权重
    # strict=False 是因为 LoRA 模型可能包含了 base model 的 key，或者 state_dict 只有 adapter 权重
    # peft 会自动处理匹配上的 adapter 权重
    missing, unexpected = pipeline.generator.load_state_dict(new_state_dict, strict=False)
    print(f"LoRA weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    # 打印一些 missing keys 以供调试 (通常 base model 的 keys 会 missing，这是正常的，只要 lora_A/lora_B 没 missing 就行)
else:
    raise FileNotFoundError(f"LoRA checkpoint not found at {lora_path}")

# --- 6. 模型移动到 GPU ---
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)

# --- 7. 数据集与推理循环 (保持不变) ---
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)

num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)
if dist.is_initialized():
    dist.barrier()

for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]

    all_video = []
    
    if args.i2v:
        prompt = batch['prompts'][0]
        prompts = [prompt] * args.num_samples
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)
        sampled_noise = torch.randn([args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16)
    else:
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        prompts = [extended_prompt] * args.num_samples if extended_prompt else [prompt] * args.num_samples
        initial_latent = None
        sampled_noise = torch.randn([args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16)

    # Inference
    # 注意：pipeline.inference 内部调用的是 self.generator(...)
    # 这里的 self.generator 已经是被 get_peft_model 包装过的了，会自动触发 LoRA 计算
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        low_memory=low_memory,
    )
    
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    video_out = 255.0 * torch.cat(all_video, dim=1)
    pipeline.vae.model.clear_cache()

    if idx < num_prompts:
        for seed_idx in range(args.num_samples):
            filename = f'{idx}-{seed_idx}_lora.mp4' if args.save_with_index else f'{prompt[:100]}-{seed_idx}_lora.mp4'
            output_path = os.path.join(args.output_folder, filename)
            write_video(output_path, video_out[seed_idx], fps=16)

print("Inference finished.")