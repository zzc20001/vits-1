import os
import json
import argparse
import itertools
import math
import logging
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

# 导入VITS核心模块
import commons
import utils
from data_utils import (
  TextAudioLoader,
  TextAudioCollate
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from text.symbols import symbols

# ===================== 全局配置（适配4GB显存） =====================
torch.backends.cudnn.benchmark = True
global_step = 0

# 低秩微调配置
LOW_RANK_CKPT_PATH = "./init_low_rank_model.pth"  # 低秩权重路径
FREEZE_NON_LOW_RANK = True                        # 冻结非低秩层
EXPECTED_LOW_RANK_PARAMS = 54                     # 预期低秩参数数量

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===================== 工具函数 =====================
def hparams_to_dict(hparams_obj):
    """将HParams对象转为字典"""
    if hasattr(hparams_obj, '__dict__'):
        return hparams_obj.__dict__.copy()
    return {}

def freeze_non_low_rank_params(net_g):
    """冻结非低秩层参数（仅保留W1/W2可训练）"""
    frozen_count = 0
    trainable_count = 0
    for name, param in net_g.named_parameters():
        # 仅保留低秩层参数可训练
        if (
            "enc_p.encoder.attn_layers" in name
            and ("conv_q" in name or "conv_k" in name or "conv_v" in name)
            and ("W1" in name or "W2" in name)
        ):
            param.requires_grad = True
            trainable_count += 1
            logger.info(f"✅ 可训练参数：{name} | 形状：{param.shape}")
        else:
            param.requires_grad = False
            frozen_count += 1
    
    logger.info(f"\n📊 参数冻结统计：")
    logger.info(f"   可训练参数数量：{trainable_count}（预期{EXPECTED_LOW_RANK_PARAMS}）")
    logger.info(f"   冻结参数数量：{frozen_count}")
    if trainable_count != EXPECTED_LOW_RANK_PARAMS:
        logger.warning(f"⚠️  可训练参数数量不符！预期{EXPECTED_LOW_RANK_PARAMS}，实际{trainable_count}")
    
    return trainable_count

def evaluate(hps, generator, eval_loader, writer_eval, global_step):
    """轻量化验证（适配4GB显存）"""
    generator.eval()
    device = next(generator.parameters()).device
    with torch.no_grad():
        # 仅取第一个样本验证（节省显存）
        for batch in eval_loader:
            x, x_lengths, spec, spec_lengths, y, y_lengths = batch
            x = x[:1].to(device)
            x_lengths = x_lengths[:1].to(device)
            spec = spec[:1].to(device)
            spec_lengths = spec_lengths[:1].to(device)
            y = y[:1].to(device)
            y_lengths = y_lengths[:1].to(device)
            break
        
        # 推理（减小最大长度）
        y_hat, attn, mask, *_ = generator.infer(x, x_lengths, max_len=500)
        y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

        # 计算Mel谱图
        mel = spec_to_mel_torch(
            spec, 
            hps.data.filter_length, 
            hps.data.n_mel_channels, 
            hps.data.sampling_rate,
            hps.data.mel_fmin, 
            hps.data.mel_fmax)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )

    # 记录结果
    image_dict = {
        "eval/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
        "eval/mel_gt": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
    }
    audio_dict = {
        "eval/audio_gen": y_hat[0,:,:y_hat_lengths[0]],
        "eval/audio_gt": y[0,:,:y_lengths[0]]
    }
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

# ===================== 主训练函数（单进程，适配4GB显存） =====================
def main():
    """单进程单GPU训练（禁用分布式，适配4GB显存）"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    # 加载配置
    hps = utils.get_hparams()
    
    # 4GB显存强制优化配置
    hps.train.batch_size = 1               # 最小批次
    hps.train.fp16_run = True              # 开启FP16
    hps.train.segment_size = 4096          # 最小音频片段长度
    hps.train.log_interval = 5             # 更频繁的日志输出
    hps.train.eval_interval = 50           # 验证间隔
    hps.train.epochs = 50                  # 低秩微调轮次

    # 设备初始化
    device = torch.device("cuda:0")
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(device)
    logger.info(f"===== VITS低秩微调训练（4GB显存优化） =====")
    logger.info(f"设备：{device} | 批次大小：{hps.train.batch_size} | FP16：{hps.train.fp16_run}")
    logger.info(f"音频片段长度：{hps.train.segment_size} | 训练轮次：{hps.train.epochs}")

    # 日志和TensorBoard
    logger.info(f"模型保存路径：{hps.model_dir}")
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    # ===================== 1. 数据加载（单进程，禁用worker） =====================
# 训练代码中main函数里的DataLoader构建部分（替换原有）
# ===================== 1. 数据加载（单进程，禁用worker） =====================
# 训练集
    train_dataset = TextAudioLoader(hps.data.training_files, hps)
# 实例化Collate类（自动适配结构）
    collate_fn = TextAudioCollate(return_ids=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        shuffle=True,
        num_workers=0,          # 禁用worker进程（关键修复）
        pin_memory=False,       # 关闭pin_memory
        collate_fn=collate_fn,
        drop_last=True
    )

# 验证集
    eval_dataset = TextAudioLoader(hps.data.validation_files, hps)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=False
    )

    # ===================== 2. 模型构建 + 低秩Encoder替换 =====================
    logger.info("构建SynthesizerTrn模型...")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)

    # 替换为低秩Encoder（核心）
    logger.info("替换enc_p.encoder为低秩版本...")
    from attentions import Encoder as LowRankEncoder
    model_cfg = hparams_to_dict(hps.model)
    encoder_config = {
        "hidden_channels": model_cfg.get("hidden_channels", 192),
        "filter_channels": model_cfg.get("filter_channels", 768),
        "n_heads": model_cfg.get("n_heads", 2),
        "n_layers": model_cfg.get("n_layers", 6),
        "kernel_size": model_cfg.get("kernel_size", 3),
        "p_dropout": model_cfg.get("p_dropout", 0.1),
        "window_size": model_cfg.get("window_size", 4)
    }
    low_rank_encoder = LowRankEncoder(**encoder_config).to(device)
    net_g.enc_p.encoder = low_rank_encoder
    logger.info("✅ 低秩Encoder替换完成")

    # 加载低秩权重
    if os.path.exists(LOW_RANK_CKPT_PATH):
        logger.info(f"加载低秩权重：{LOW_RANK_CKPT_PATH}")
        low_rank_ckpt = torch.load(
            LOW_RANK_CKPT_PATH,
            map_location=device,
            weights_only=True
        )
        # 严格=False，忽略不匹配的参数
        net_g.load_state_dict(low_rank_ckpt.get("net_g", low_rank_ckpt), strict=False)
        logger.info("✅ 低秩权重加载成功")
    else:
        raise FileNotFoundError(f"❌ 低秩权重文件不存在：{LOW_RANK_CKPT_PATH}")

    # 构建判别器（冻结，减少显存占用）
    logger.info("构建判别器并冻结...")
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
    for param in net_d.parameters():
        param.requires_grad = False
    logger.info("✅ 判别器已冻结")

    # ===================== 3. 参数冻结 + 优化器配置 =====================
    # 冻结非低秩层
    if FREEZE_NON_LOW_RANK:
        logger.info("冻结非低秩层参数...")
        trainable_count = freeze_non_low_rank_params(net_g)
        if trainable_count == 0:
            raise ValueError("❌ 无可用的可训练参数！请检查低秩层定义")

    # 仅优化可训练参数（低秩层）
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        lr=1e-5,                # 低秩微调小学习率
        betas=hps.train.betas,
        eps=hps.train.eps,
        weight_decay=1e-6       # 小权重衰减
    )

    # 学习率调度器（慢衰减）
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g,
        gamma=0.99,             # 更慢的衰减率
        last_epoch=-1
    )

    # FP16梯度缩放（核心显存优化）
    scaler = GradScaler(enabled=hps.train.fp16_run)

    # ===================== 4. 核心训练循环 =====================
    global global_step
    global_step = 0
    total_loss = 0.0

    logger.info("开始低秩微调训练...")
    for epoch in range(1, hps.train.epochs + 1):
        logger.info(f"\n===== Epoch {epoch}/{hps.train.epochs} =====")
        net_g.train()
        net_d.train()

        for batch_idx, batch in enumerate(train_loader):
            # 数据解包并移到GPU
            x, x_lengths, spec, spec_lengths, y, y_lengths = batch
            x = x.to(device, non_blocking=True)
            x_lengths = x_lengths.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            spec_lengths = spec_lengths.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_lengths = y_lengths.to(device, non_blocking=True)

            # FP16前向传播
            with autocast(enabled=hps.train.fp16_run):
                # 生成器前向计算
                y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths)

                # 计算Mel谱图
                mel = spec_to_mel_torch(
                    spec, 
                    hps.data.filter_length, 
                    hps.data.n_mel_channels, 
                    hps.data.sampling_rate,
                    hps.data.mel_fmin, 
                    hps.data.mel_fmax)
                y_mel = commons.slice_segments(
                    mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1), 
                    hps.data.filter_length, 
                    hps.data.n_mel_channels, 
                    hps.data.sampling_rate, 
                    hps.data.hop_length, 
                    hps.data.win_length, 
                    hps.data.mel_fmin, 
                    hps.data.mel_fmax
                )

                # 音频切片
                y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

                # 判别器计算（仅前向，冻结参数）
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)

                # 生成器损失计算
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                
                # 总损失
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

            # 反向传播
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            
            # 梯度裁剪（防止爆炸）
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), 1.0)
            
            # 更新优化器
            scaler.step(optim_g)
            scaler.update()

            # 损失累计
            total_loss += loss_gen_all.item()
            global_step += 1

            # ===================== 日志输出 & 验证 & 保存 =====================
            if global_step % hps.train.log_interval == 0:
                # 计算平均损失
                avg_loss = total_loss / hps.train.log_interval
                lr = optim_g.param_groups[0]['lr']
                
                # 打印日志
                logger.info(f"Batch {batch_idx} | Loss: {avg_loss:.4f} | LR: {lr:.6f} | Grad Norm: {grad_norm_g:.4f}")
                
                # TensorBoard记录
                scalar_dict = {
                    "loss/g/total": avg_loss,
                    "loss/g/mel": loss_mel.item(),
                    "loss/g/kl": loss_kl.item(),
                    "loss/g/fm": loss_fm.item(),
                    "learning_rate": lr,
                    "grad_norm_g": grad_norm_g
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict,
                    images={
                        "mel/gt": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                        "mel/gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                        "attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
                    }
                )
                total_loss = 0.0

            # 验证和保存权重
            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval, global_step)
                save_path = os.path.join(hps.model_dir, f"G_lowrank_{global_step}.pth")
                utils.save_checkpoint(
                    net_g, optim_g, lr, epoch, save_path
                )
                logger.info(f"✅ 权重已保存：{save_path}")

        # 学习率衰减
        scheduler_g.step()
        logger.info(f"Epoch {epoch} 完成 | 当前学习率：{optim_g.param_groups[0]['lr']:.6f}")

    # ===================== 训练完成 =====================
    final_path = os.path.join(hps.model_dir, "G_lowrank_final.pth")
    utils.save_checkpoint(
        net_g, optim_g, optim_g.param_groups[0]['lr'], hps.train.epochs, final_path
    )
    logger.info(f"🎉 低秩微调训练完成！最终权重保存：{final_path}")

# ===================== 入口函数 =====================
if __name__ == "__main__":
    main()