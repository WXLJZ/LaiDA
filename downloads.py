# --coding:utf-8--
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2-7B-Instruct', cache_dir="/hy-tmp/models")