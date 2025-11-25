from tqdm import tqdm
import numpy as np
import argparse
import torch
import lmdb
import glob
import os

# 只保留 store_arrays_to_lmdb，不再导入 process_data_dict
from utils.lmdb import store_arrays_to_lmdb 

def main():
    """
    Aggregate all ode pairs inside a folder into a lmdb dataset.
    Each pt file should contain a (key, value) pair representing a
    video's ODE trajectories.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        required=True, help="path to ode pairs")
    parser.add_argument("--lmdb_path", type=str,
                        required=True, help="path to lmdb")

    args = parser.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.data_path, "*.pt")))

    # figure out the maximum map size needed
    total_array_size = 5000000000000  # 5TB

    env = lmdb.open(args.lmdb_path, map_size=total_array_size * 2)

    counter = 0
    seen_prompts = set()  # for deduplication
    
    # 用于记录最后处理的一个 dict，以便在循环结束后保存 shape
    last_processed_dict = None

    for index, file in tqdm(enumerate(all_files)):

        # === 【修改】添加这一段用于快速测试 ===
        if index >= 50:  # 只需要 50 条数据用于测试
            print("Debug Mode: Reached 50 items, stopping early to save LMDB...")
            break
        # ====================================
        
        try:
            # 1. 强制加载到 CPU
            data_dict = torch.load(file, map_location="cpu")
        except Exception as e:
            print(f"[Error] Failed to load {file}: {e}")
            continue

        # === 手动处理数据 (替代 process_data_dict) ===
        
        # 2. Prompt 去重逻辑
        if 'prompts' in data_dict:
            prompt = data_dict['prompts']
            # 如果 prompt 是列表，取第一个元素
            if isinstance(prompt, list):
                prompt = prompt[0]
            
            # 转成字符串进行哈希去重
            prompt_str = str(prompt)
            if prompt_str in seen_prompts:
                # print(f"Skipping duplicate prompt in {file}")
                continue
            seen_prompts.add(prompt_str)
        
        # 3. Tensor 转 Numpy FP16 (核心修复)
        # 我们手动找到 ode_latent 并转换，不再依赖那个报错的函数
        if 'ode_latent' in data_dict:
            tensor_data = data_dict['ode_latent']
            
            # 容错：防止 ode_latent 依然包裹在字典里
            if isinstance(tensor_data, dict):
                print(f"[Warning] nested dict found in {file}, extracting...")
                if 'ode_latent' in tensor_data:
                    tensor_data = tensor_data['ode_latent']
                else:
                    tensor_data = list(tensor_data.values())[0]

            # 确保是 Tensor 后进行转换
            if torch.is_tensor(tensor_data):
                # 转为半精度 (fp16) 并转为 numpy array，节省空间
                data_dict['ode_latent'] = tensor_data.half().numpy()
            elif isinstance(tensor_data, np.ndarray):
                # 已经是 numpy 了
                pass
            else:
                print(f"[Error] Unknown data type in {file}: {type(tensor_data)}")
                continue
        else:
            print(f"[Error] No 'ode_latent' key in {file}")
            continue

        # === 处理结束 ===

        # 4. 写入 LMDB
        store_arrays_to_lmdb(env, data_dict, start_index=counter)
        counter += 1 # 这里的 counter 逻辑稍微改一下，通常是一个样本占一个 index
        # 注意：如果 store_arrays_to_lmdb 内部逻辑是一个样本占一行，这里+1
        # 如果你之前的逻辑是 counter += len(prompts)，请根据实际情况调整，通常 batch=1 时就是 +1
        
        last_processed_dict = data_dict

    # 5. 保存 shape 信息 (使用最后一个成功处理的数据)
    if last_processed_dict is not None:
        print("Saving shape info to LMDB...")
        with env.begin(write=True) as txn:
            for key, val in last_processed_dict.items():
                # 只保存数组类型的 shape
                if isinstance(val, np.ndarray):
                    print(f"Key: {key}, Shape: {val.shape}")
                    array_shape = np.array(val.shape)
                    #这行通常是为了记录总数，但如果只存shape，不需要修改dim0
                    # array_shape[0] = counter 
                    
                    shape_key = f"{key}_shape".encode()
                    shape_str = " ".join(map(str, array_shape))
                    txn.put(shape_key, shape_str.encode())
                
            # 单独保存一下总长度
            txn.put("length".encode(), str(counter).encode())

    print(f"Done. Total processed: {counter}")

if __name__ == "__main__":
    main()