import importlib.metadata
import sys

# 指定要忽略的包（可选）
IGNORE_LIST = {'pip', 'setuptools', 'wheel', 'distribute'}

try:
    with open("requirements.txt", "w") as f:
        dists = list(importlib.metadata.distributions())
        # 按名称排序，好看一点
        dists.sort(key=lambda x: x.metadata['Name'].lower())
        
        for dist in dists:
            name = dist.metadata['Name']
            version = dist.version
            if name not in IGNORE_LIST:
                f.write(f"{name}=={version}\n")
                
except Exception as e:
    print(f"❌ 生成失败: {e}")