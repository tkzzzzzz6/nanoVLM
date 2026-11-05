"""
Colab 会话一次性初始化脚本（01 - Setup & Resident Cache）

核心职责：
1. 解压本地同步上来的 nanoVLM 项目代码包（/content/nanovlm_src.tar.gz -> /content/nanoVLM）；
2. 拉取预训练视觉编码器（SigLIP2）与语言模型（SmolLM2）权重并加载进云端 GPU 显存；
3. 备份一份干净的初始模型权重（CPU RAM）以便后续多次重跑 train.py 时 0.1 秒快速重置；
4. 拉取并切分训练/验证数据集，将 model、tokenizer、image_processor、数据集统一缓存在
   远程 Jupyter Kernel 的全局变量 `COLAB_ENV` 中，供 `colab/train.py` 反复秒级复用。
"""

import os
import sys
import time
import tarfile
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore", message=".*Length of IterableDataset.*")

import PIL.PngImagePlugin
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

import torch
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names

REMOTE_PROJECT_DIR = "/content/nanoVLM"
SRC_ARCHIVE_PATH = "/content/nanovlm_src.tar.gz"


def sync_project_workspace() -> None:
    """若检测到本地上传了新的源码压缩包，自动解压并挂载到 sys.path。"""
    os.makedirs(REMOTE_PROJECT_DIR, exist_ok=True)
    if os.path.isfile(SRC_ARCHIVE_PATH):
        mtime = os.path.getmtime(SRC_ARCHIVE_PATH)
        last_mtime = globals().get("_NANOVLM_ARCHIVE_MTIME")
        if last_mtime != mtime:
            print(f"[setup] 解压最新项目源码包 {SRC_ARCHIVE_PATH} -> {REMOTE_PROJECT_DIR} ...")
            with tarfile.open(SRC_ARCHIVE_PATH, "r:gz") as tar:
                tar.extractall(REMOTE_PROJECT_DIR)
            globals()["_NANOVLM_ARCHIVE_MTIME"] = mtime

    os.chdir(REMOTE_PROJECT_DIR)
    if REMOTE_PROJECT_DIR not in sys.path:
        sys.path.insert(0, REMOTE_PROJECT_DIR)


def setup_colab_environment() -> dict:
    t0 = time.time()
    sync_project_workspace()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("[setup] Hugging Face Token 鉴权完成。")

    import models.config as config
    from data.processors import get_image_processor, get_tokenizer
    from models.vision_language_model import VisionLanguageModel

    # 检测硬件与最佳混合精度类型（L4/A100/H100 用 bfloat16，T4 自动用 float16）
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"[setup] 检测到云端 GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) | 混合精度: {amp_dtype}")
    else:
        device = torch.device("cpu")
        amp_dtype = torch.float32
        print("[setup] 警告: 当前为 CPU 环境，训练速度会较慢。")

    # 构建适合单卡 Colab 的模型配置（支持通过环境变量覆盖）
    vlm_cfg = config.VLMConfig()
    vlm_cfg.vit_model_type = os.getenv("VIT_MODEL_TYPE", vlm_cfg.vit_model_type)
    vlm_cfg.lm_model_type = os.getenv("LM_MODEL_TYPE", "HuggingFaceTB/SmolLM2-360M-Instruct")
    vlm_cfg.lm_tokenizer = os.getenv("LM_TOKENIZER", "HuggingFaceTB/SmolLM2-360M-Instruct")
    vlm_cfg.max_img_size = int(os.getenv("MAX_IMG_SIZE", "1024"))
    vlm_cfg.lm_max_length = int(os.getenv("LM_MAX_LENGTH", "1024"))
    vlm_cfg.vlm_checkpoint_path = os.getenv("CHECKPOINT_DIR", "checkpoints")
    vlm_cfg.hf_repo_name = os.getenv("HF_REPO_NAME") or None

    print(f"[setup] 正在拉取并初始化骨干模型: ViT={vlm_cfg.vit_model_type} | LM={vlm_cfg.lm_model_type} ...")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    model = VisionLanguageModel(vlm_cfg, load_backbone=True)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] nanoVLM 模型加载完成，总参数量: {total_params:,}")

    # 在 CPU 内存保存一份初始预训练权重快照，方便后续多次跑 train.py 时秒级重置参数
    init_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # 初始化 Tokenizer 与图像处理器
    image_processor = get_image_processor(
        vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len
    )
    tokenizer = get_tokenizer(
        vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template
    )

    # 拉取并缓存数据集（默认采用 Colab 友好的 the_cauldron/tqa，也支持切换 FineVision 流式数据集）
    dataset_path = os.getenv("DATASET_PATH", "HuggingFaceM4/the_cauldron")
    default_subset = "default" if "FineVision" in dataset_path else "tqa"
    dataset_names_raw = os.getenv("DATASET_NAMES", default_subset)
    dataset_names = [x.strip() for x in dataset_names_raw.split(",") if x.strip()]
    if "all" in dataset_names:
        dataset_names = get_dataset_config_names(dataset_path)

    default_stream = "1" if "FineVision" in dataset_path else "0"
    stream_dataset = os.getenv("STREAM_DATASET", default_stream) == "1"
    val_size = int(os.getenv("VAL_SIZE", "64"))
    data_cutoff = int(os.getenv("DATA_CUTOFF", "512"))  # 设为 0 表示不截断全量使用

    print(
        f"[setup] 正在拉取数据集: {dataset_path} | 子集={dataset_names} | streaming={stream_dataset} ..."
    )
    combined_train_data = []
    for ds_name in dataset_names:
        ds = load_dataset(
            dataset_path, ds_name, streaming=stream_dataset, on_bad_files="warn"
        )["train"]
        # 触发首条样本校验连通性
        _ = next(iter(ds)) if stream_dataset else ds[0]
        combined_train_data.append(ds)

    raw_ds = (
        concatenate_datasets(combined_train_data)
        if len(combined_train_data) > 1
        else combined_train_data[0]
    )

    if stream_dataset:
        val_ds = raw_ds.take(val_size)
        train_ds = raw_ds.skip(val_size)
        print(f"[setup] 流式数据集就绪 (val_size={val_size})。")
    else:
        raw_ds = raw_ds.shuffle(seed=0)
        total_samples = len(raw_ds) if data_cutoff <= 0 else min(len(raw_ds), data_cutoff)
        actual_val = min(val_size, max(1, int(total_samples * 0.2)))
        val_ds = raw_ds.select(range(actual_val))
        train_ds = raw_ds.select(range(actual_val, total_samples))
        print(
            f"[setup] 内存数据集就绪: train={len(train_ds)} 条 | val={len(val_ds)} 条 (总截取={total_samples})。"
        )

    colab_env = {
        "model": model,
        "init_state_dict": init_state_dict,
        "vlm_cfg": vlm_cfg,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "dataset_path": dataset_path,
        "dataset_names": tuple(dataset_names),
        "stream_dataset": stream_dataset,
        "device": device,
        "amp_dtype": amp_dtype,
    }
    print(f"[setup] 环境预热全部完成，耗时 {time.time() - t0:.1f}s！全局变量 `COLAB_ENV` 已常驻内存。")
    return colab_env


COLAB_ENV = setup_colab_environment()
