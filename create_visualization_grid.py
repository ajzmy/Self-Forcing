# 文件: create_visualization_grid.py (版本6 - 最终优化版)
# - 改为抽取每个Block的“中间帧”
# - 修正了 renoised 模式下不显示 Initial Noise 的问题

import os
import glob
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

# --- 参数 ---
NUM_COLS_PER_BLOCK = 1
PADDING = 40
HEADER_SIZE = 100
try:
    FONT = ImageFont.truetype("arial.ttf", 20)
    LABEL_FONT = ImageFont.truetype("arialbd.ttf", 24)
except IOError:
    FONT = ImageFont.load_default()
    LABEL_FONT = FONT
# --- 参数结束 ---

def create_visualization_grid(input_folder, output_image, viz_type):
    """
    根据指定的类型 (denoised 或 renoised) 创建可视化网格图。
    """
    print(f"[*] Creating visualization of type: '{viz_type}'")
    print(f"[*] Searching for intermediate videos in: {input_folder}")
    
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    if not video_files:
        print(f"[!] Error: No .mp4 files found in {input_folder}")
        return

    # 1. 解析并根据 viz_type 筛选文件
    structured_files = []
    for f in video_files:
        basename = os.path.basename(f).replace('.mp4', '')
        try:
            is_renoised = "_renoised" in basename
            clean_basename = basename.replace('_renoised', '')
            parts = clean_basename.split('_')
            
            step_num = int(parts[5])
            block_num = int(parts[1])
            
            # --- 核心筛选逻辑 (已修正) ---
            # Initial Noise (step < 0) 在两种模式下都应该被包括
            if step_num < 0:
                pass # 总是保留 Initial Noise
            elif viz_type == 'denoised' and is_renoised:
                continue
            elif viz_type == 'renoised' and not is_renoised:
                continue
            # --- 筛选结束 ---

            structured_files.append({
                'path': f, 
                'block': block_num,
                'step': step_num,
            })
        except (IndexError, ValueError):
            print(f"[!] Warning: Skipping malformed filename: {f}")
            continue
    
    if not structured_files:
        print(f"[!] Error: No valid video files found for the type '{viz_type}'.")
        return
        
    # 2. 构建 grid_rows_data
    grid_rows_data = []
    structured_files.sort(key=lambda x: x['step'])
    
    grouped_by_step = defaultdict(list)
    for f in structured_files:
        grouped_by_step[f['step']].append(f)

    for step_num in sorted(grouped_by_step.keys()):
        grouped_by_step[step_num].sort(key=lambda x: x['block'])
        
        if step_num < 0:
            label = "Initial Noise"
        else:
            if viz_type == 'denoised':
                label = f"Pass {step_num + 1}: Denoised Pred"
            else:
                label = f"Pass {step_num + 1}: Re-noised Input"

        grid_rows_data.append({'label': label, 'files': grouped_by_step[step_num]})
    
    # 3. 计算画布尺寸和绘制
    num_rows = len(grid_rows_data)
    num_blocks = max(len(row['files']) for row in grid_rows_data) if grid_rows_data else 0
    num_cols_total = NUM_COLS_PER_BLOCK * num_blocks
    print(f"[*] Grid will have {num_rows} rows and {num_cols_total} columns (frames).")
    
    first_video_path = grid_rows_data[0]['files'][0]['path']
    cap = cv2.VideoCapture(first_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    canvas_width = HEADER_SIZE + (frame_width + PADDING) * num_cols_total
    canvas_height = HEADER_SIZE + (frame_height + PADDING) * num_rows
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    main_title = "Denoised Predictions per Pass" if viz_type == 'denoised' else "Re-noised Inputs for Next Pass"
    draw.text((canvas_width / 2 - 200, PADDING // 2), main_title, font=LABEL_FONT, fill="black")
    draw.text((HEADER_SIZE + (canvas_width - HEADER_SIZE) / 2 - 150, PADDING // 2 + 30), "Video Frame Sequence ->", font=FONT, fill="black")
    draw.text((PADDING, PADDING // 2), "Denoising Process ->", font=FONT, fill="black")

    for i, row_data in enumerate(grid_rows_data):
        draw.text((PADDING, HEADER_SIZE + i * (frame_height + PADDING) + frame_height // 2 - 15), row_data['label'], font=FONT, fill="black")
        
        for block_idx, file_info in enumerate(row_data['files']):
            if i == 0:
                block_label_x = HEADER_SIZE + block_idx * NUM_COLS_PER_BLOCK * (frame_width + PADDING)
                draw.text((block_label_x, HEADER_SIZE - PADDING), f"Block {block_idx}", font=FONT, fill="gray")
                draw.line([(block_label_x - PADDING//2, HEADER_SIZE - 5), (block_label_x - PADDING//2, canvas_height)], fill="lightgray", width=1)
            
            cap = cv2.VideoCapture(file_info['path'])
            if not cap.isOpened(): continue
            
            # --- 核心修改：改回抽取“中间帧” ---
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices_to_capture = [total_frames // 2]
            # --- 修改结束 ---
            
            for j, frame_idx in enumerate(frame_indices_to_capture):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    total_col_idx = block_idx * NUM_COLS_PER_BLOCK + j
                    paste_x = HEADER_SIZE + total_col_idx * (frame_width + PADDING)
                    paste_y = HEADER_SIZE + i * (frame_height + PADDING)
                    canvas.paste(pil_img, (paste_x, paste_y))
            cap.release()

    canvas.save(output_image)
    print(f"[+] Successfully created and saved visualization grid to: {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a visualization grid from intermediate diffusion videos.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing the intermediate video steps.")
    parser.add_argument("--output_image", type=str, required=True, help="Path to save the final grid image.")
    parser.add_argument("--type", type=str, choices=['denoised', 'renoised'], default='denoised', 
                        help="Type of visualization to create: 'denoised' for predictions, 'renoised' for re-noised inputs.")
    args = parser.parse_args()
    
    create_visualization_grid(args.input_folder, args.output_image, args.type)