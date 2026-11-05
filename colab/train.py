"""
Colab 增量训练与调参脚本（02 - Fast Iterative Training）

核心特点：
1. 直接复用 `colab/setup.py` 常驻在云端显存/内存中的 `COLAB_ENV`（免去每次重跑重新下载模型与数据集）；
2. 默认从内存快照 0.1 秒重置初始预训练权重（`RESET_WEIGHTS=1`），保证每次改学习率/超参实验互不污染；
   若希望在上一轮训练结果上继续微调，只需传 `RESET_WEIGHTS=0`；
3. 自动适配 T4（FP16 + GradScaler）与 L4/A100/H100（BF16），训练结束保存最优权重并绘制 Loss 曲线。
"""

import importlib
import math
import os
import random
import sys
import tarfile
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

REMOTE_PROJECT_DIR = "/content/nanoVLM"
SRC_ARCHIVE_PATH = "/content/nanovlm_src.tar.gz"


def refresh_local_code_if_updated() -> None:
    """如果通过 run_colab.sh sync 上传了新代码包，自动解压并热重载本地模块。"""
    if os.path.isfile(SRC_ARCHIVE_PATH):
        mtime = os.path.getmtime(SRC_ARCHIVE_PATH)
        if globals().get("_NANOVLM_ARCHIVE_MTIME") != mtime:
            print("[train] 检测到更新的源码包，正在热更新 /content/nanoVLM ...")
            with tarfile.open(SRC_ARCHIVE_PATH, "r:gz") as tar:
                tar.extractall(REMOTE_PROJECT_DIR)
            globals()["_NANOVLM_ARCHIVE_MTIME"] = mtime
            for mod_name in list(sys.modules.keys()):
                if mod_name.startswith(("models.", "data.", "utils.")):
                    importlib.reload(sys.modules[mod_name])

    if os.path.isdir(REMOTE_PROJECT_DIR):
        os.chdir(REMOTE_PROJECT_DIR)
        if REMOTE_PROJECT_DIR not in sys.path:
            sys.path.insert(0, REMOTE_PROJECT_DIR)


refresh_local_code_if_updated()

# 若用户重启过 Kernel 或尚未运行 setup.py，自动回退执行 setup.py
if "COLAB_ENV" not in globals():
    setup_script = os.path.join(REMOTE_PROJECT_DIR, "colab", "setup.py")
    if os.path.isfile(setup_script):
        print("[train] 未检测到内存中的 COLAB_ENV，正在自动执行 colab/setup.py 初始化...")
        with open(setup_script, "r", encoding="utf-8") as f:
            exec(compile(f.read(), setup_script, "exec"), globals())
    else:
        raise RuntimeError("请先运行 `./colab/run_colab.sh setup` 完成云端环境与模型预加载！")

import models.config as config
from data.advanced_datasets import ConstantLengthDataset
from data.collators import VQACollator
from data.data_utils import synchronized_dataloader_step
from data.datasets import VQADataset


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_lr(step: int, max_lr: float, max_steps: int) -> float:
    min_lr = max_lr * 0.1
    warmup_steps = max(1.0, max_steps * 0.03)
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1.0, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def build_colab_train_config(env: dict) -> config.TrainConfig:
    train_cfg = config.TrainConfig()
    train_cfg.train_dataset_path = env["dataset_path"]
    train_cfg.train_dataset_name = env["dataset_names"]
    train_cfg.stream_dataset = env["stream_dataset"]

    # 单卡 Colab 友好的默认超参数（均可直接改这里，或通过 run_colab.sh 环境变量传入）
    train_cfg.batch_size = int(os.getenv("BATCH_SIZE", "2"))
    train_cfg.gradient_accumulation_steps = int(os.getenv("GRAD_ACCUM", "4"))
    train_cfg.max_training_steps = int(os.getenv("MAX_STEPS", "200"))
    train_cfg.eval_interval = int(os.getenv("EVAL_INTERVAL", "50"))
    train_cfg.stats_log_interval = int(os.getenv("LOG_INTERVAL", "10"))
    train_cfg.lr_mp = float(os.getenv("LR_MP", "0.005"))
    train_cfg.lr_vision_backbone = float(os.getenv("LR_VISION", "5e-5"))
    train_cfg.lr_language_backbone = float(os.getenv("LR_LANGUAGE", "5e-5"))
    train_cfg.max_images_per_example = int(os.getenv("MAX_IMAGES_PER_EXAMPLE", "2"))
    train_cfg.max_images_per_knapsack = int(os.getenv("MAX_IMAGES_PER_KNAPSACK", "8"))
    train_cfg.max_sample_length = env["vlm_cfg"].lm_max_length

    # 禁用 Slurm 专属评测，按需开启 WandB
    train_cfg.use_lmms_eval = False
    train_cfg.log_wandb = bool(os.getenv("WANDB_API_KEY")) and os.getenv("LOG_WANDB", "0") == "1"
    if os.getenv("WANDB_ENTITY"):
        train_cfg.wandb_entity = os.getenv("WANDB_ENTITY")
    return train_cfg


def build_dataloaders(env: dict, train_cfg: config.TrainConfig):
    vlm_cfg = env["vlm_cfg"]
    tokenizer = env["tokenizer"]
    image_processor = env["image_processor"]

    train_vqa = VQADataset(
        env["train_ds"],
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
        train_cfg.relevance_min_rating,
        train_cfg.image_correspondence_min_rating,
        train_cfg.visual_dependency_min_rating,
        train_cfg.formatting_min_rating,
    )
    val_vqa = VQADataset(
        env["val_ds"],
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
        train_cfg.relevance_min_rating,
        train_cfg.image_correspondence_min_rating,
        train_cfg.visual_dependency_min_rating,
        train_cfg.formatting_min_rating,
    )

    train_packed = ConstantLengthDataset(
        train_vqa,
        infinite=False,
        max_sample_length=train_cfg.max_sample_length,
        seq_length=vlm_cfg.lm_max_length,
        num_of_sequences=train_cfg.batch_size * 4,
        queue_size=8,
        max_images_per_example=train_cfg.max_images_per_example,
        max_images_per_knapsack=train_cfg.max_images_per_knapsack,
    )
    val_packed = ConstantLengthDataset(
        val_vqa,
        infinite=False,
        max_sample_length=train_cfg.max_sample_length,
        seq_length=vlm_cfg.lm_max_length,
        num_of_sequences=train_cfg.batch_size * 4,
        queue_size=8,
        max_images_per_example=train_cfg.max_images_per_example,
        max_images_per_knapsack=train_cfg.max_images_per_knapsack,
    )

    collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    g = torch.Generator().manual_seed(0)

    # 在持久 Jupyter Kernel 中使用 num_workers=0 可避免多进程管道跨次执行挂起
    num_workers = int(os.getenv("NUM_WORKERS", "0"))
    train_loader = DataLoader(
        train_packed,
        batch_size=train_cfg.batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_packed,
        batch_size=train_cfg.batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return train_loader, val_loader


def evaluate(model, val_loader, device, amp_dtype, max_val_batches: int = 32) -> float:
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in synchronized_dataloader_step(iter(val_loader), False):
            if batches >= max_val_batches:
                break
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
            total_loss += loss.item()
            batches += 1

    model.train()
    return total_loss / max(1, batches)


def run_training() -> None:
    env = COLAB_ENV
    model = env["model"]
    vlm_cfg = env["vlm_cfg"]
    device = env["device"]
    amp_dtype = env["amp_dtype"]

    # 默认秒级恢复初始预训练权重，确保每次重跑实验起点一致；若传 RESET_WEIGHTS=0 则接着上一次权重继续训
    if os.getenv("RESET_WEIGHTS", "1") == "1":
        t_reset = time.time()
        model.load_state_dict(env["init_state_dict"])
        model.to(device)
        print(f"[train] 已从内存快照重置初始预训练权重 ({time.time() - t_reset:.2f}s)。")
    else:
        print("[train] RESET_WEIGHTS=0，保留上一轮显存权重继续训练。")

    train_cfg = build_colab_train_config(env)
    train_loader, val_loader = build_dataloaders(env, train_cfg)

    eff_bs = train_cfg.batch_size * train_cfg.gradient_accumulation_steps
    run_name = (
        f"colab_nanoVLM_bs{eff_bs}_steps{train_cfg.max_training_steps}_"
        f"lr{train_cfg.lr_mp}_{time.strftime('%m%d-%H%M%S')}"
    )
    print(
        f"[train] 开始运行: {run_name}\n"
        f"        Batch={train_cfg.batch_size} x GradAccum={train_cfg.gradient_accumulation_steps} (等效 BS={eff_bs}) | "
        f"MaxSteps={train_cfg.max_training_steps} | EvalInterval={train_cfg.eval_interval}"
    )

    wandb_run = None
    if train_cfg.log_wandb:
        import wandb
        wandb_run = wandb.init(
            entity=train_cfg.wandb_entity,
            project="nanoVLM",
            name=run_name,
            config={"VLMConfig": asdict(vlm_cfg), "TrainConfig": asdict(train_cfg)},
        )

    param_groups = []
    for module, lr in [
        (model.MP, train_cfg.lr_mp),
        (model.vision_encoder, train_cfg.lr_vision_backbone),
        (model.decoder, train_cfg.lr_language_backbone),
    ]:
        params = list(module.parameters())
        if lr > 0:
            for p in params:
                p.requires_grad = True
            param_groups.append({"params": params, "lr": lr})
        else:
            for p in params:
                p.requires_grad = False

    optimizer = optim.AdamW(param_groups)
    all_params = [p for group in optimizer.param_groups for p in group["params"]]
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_val_loss = float("inf")
    best_ckpt_dir = os.path.join(vlm_cfg.vlm_checkpoint_path, "latest_best")
    os.makedirs(vlm_cfg.vlm_checkpoint_path, exist_ok=True)

    global_step = 0
    epoch = 0
    step_losses, train_steps = [], []
    val_losses, val_steps = [], []
    t_start = time.time()

    model.train()
    optimizer.zero_grad()

    while global_step < train_cfg.max_training_steps:
        epoch += 1
        for i, batch in enumerate(synchronized_dataloader_step(iter(train_loader), False)):
            is_update_step = (i + 1) % train_cfg.gradient_accumulation_steps == 0
            t_batch = time.time()

            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
                scaled_loss = loss / train_cfg.gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()

            if is_update_step:
                scaler.unscale_(optimizer)
                if train_cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=train_cfg.max_grad_norm)

                pg_idx = 0
                for base_lr in (
                    train_cfg.lr_mp,
                    train_cfg.lr_vision_backbone,
                    train_cfg.lr_language_backbone,
                ):
                    if base_lr > 0:
                        optimizer.param_groups[pg_idx]["lr"] = get_lr(
                            global_step, base_lr, train_cfg.max_training_steps
                        )
                        pg_idx += 1

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                global_step += 1
                raw_loss = loss.item()
                tokens = torch.sum(attention_mask).item()
                tok_per_s = tokens / max(1e-6, time.time() - t_batch)

                step_losses.append(raw_loss)
                train_steps.append(global_step)

                if global_step % train_cfg.stats_log_interval == 0 or global_step == 1:
                    print(
                        f"  [Step {global_step:4d}/{train_cfg.max_training_steps}] "
                        f"train_loss={raw_loss:.4f} | lr_mp={optimizer.param_groups[0]['lr']:.6f} | "
                        f"tok/s={tok_per_s:.1f}"
                    )
                    if wandb_run:
                        wandb_run.log({"batch_loss": raw_loss, "tokens_per_s": tok_per_s}, step=global_step)

                if global_step % train_cfg.eval_interval == 0 or global_step == train_cfg.max_training_steps:
                    val_loss = evaluate(model, val_loader, device, amp_dtype)
                    val_losses.append(val_loss)
                    val_steps.append(global_step)
                    improved = val_loss < best_val_loss
                    if improved:
                        best_val_loss = val_loss
                        model.save_pretrained(save_directory=best_ckpt_dir)
                    mark = " *BEST (已保存)*" if improved else ""
                    print(f"  >>> [Eval Step {global_step}] val_loss={val_loss:.4f}{mark}")
                    if wandb_run:
                        wandb_run.log({"val_loss": val_loss}, step=global_step)

                if global_step >= train_cfg.max_training_steps:
                    break

    elapsed = time.time() - t_start
    print(
        f"\n[train] 训练完成！总耗时: {elapsed:.1f}s | 最优验证集 Loss: {best_val_loss:.4f} | "
        f"权重路径: /content/nanoVLM/{best_ckpt_dir}"
    )

    if vlm_cfg.hf_repo_name and os.path.isdir(best_ckpt_dir):
        from models.vision_language_model import VisionLanguageModel
        print(f"[train] 正在推送最优模型至 Hugging Face Hub: {vlm_cfg.hf_repo_name} ...")
        VisionLanguageModel.from_pretrained(best_ckpt_dir).push_to_hub(vlm_cfg.hf_repo_name)

    if wandb_run:
        wandb_run.finish()

    # 若安装了 matplotlib，自动渲染 Loss 曲线（可被 colab exec --output-image 自动捕获到本地）
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4.5))
        plt.plot(train_steps, step_losses, alpha=0.75, label="Train Loss")
        if val_steps:
            plt.plot(val_steps, val_losses, marker="o", linewidth=2, label="Val Loss")
        plt.xlabel("Step")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"nanoVLM Training ({run_name})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


run_training()
