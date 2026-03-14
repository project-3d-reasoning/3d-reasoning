from huggingface_hub import snapshot_download

# 指定你想要保存的本地普通文件夹路径
custom_path = "./data" 

dataset_path = snapshot_download(
    repo_id="zd11024/VG-LLM-Data", 
    repo_type="dataset",
    local_dir=custom_path,        # <--- 这里指定下载到哪个普通文件夹
    local_dir_use_symlinks=False  # <--- 设为 False，直接下载实体文件，不生成软链接
)

print(f"数据已完整下载至普通文件夹: {dataset_path}")