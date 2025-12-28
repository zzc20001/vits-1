import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import logging

# ===================== 全局配置与日志 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===================== 核心配置（请确认路径正确性） =====================
LOW_RANK_CKPT_PATH = "./init_low_rank_model.pth"  # 迁移后的低秩权重路径
FREEZE_NON_LOW_RANK = True                        # 仅训练低秩层
TRAIN_LR = 1e-5                                   # 低秩层微调学习率
EXPECTED_LOW_RANK_PARAMS = 54                     # 预期的低秩参数数量（迁移输出为54）

# ===================== 导入核心模块（关键：导入低秩Encoder） =====================
try:
    from utils import get_hparams
    from models import SynthesizerTrn
    from text import symbols
    # 导入低秩版本的Encoder（attentions.py）和原始版本（用于对比）
    from attentions import Encoder as LowRankEncoder
    import attentions_original as attentions_ori
except ImportError as e:
    logger.error(f"模块导入失败：{str(e)}")
    raise ImportError("请确认attentions.py/attentions_original.py/models.py路径正确")

# ===================== 分布式训练初始化 =====================
def init_distributed(args):
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    return args

# ===================== 主训练函数（核心逻辑） =====================
def run(rank, n_gpus, hps):
    """单进程训练逻辑（支持单/多GPU）"""
    # 分布式初始化
    if n_gpus > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:54321',
            world_size=n_gpus,
            rank=rank
        )
    
    # 设置设备
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    logger.info(f"[Rank {rank}] 使用设备：{device}")

    # ===================== 1. 构建基础模型 =====================
    logger.info(f"[Rank {rank}] 构建SynthesizerTrn基础模型...")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)

    # ===================== 2. 强制替换为低秩Encoder（核心修复） =====================
    logger.info(f"[Rank {rank}] 替换enc_p.encoder为低秩版本...")
    # 提取Encoder配置参数（与权重迁移代码完全一致）
    model_cfg = hps.model

    encoder_config = {
        "hidden_channels": model_cfg["hidden_channels"],
        "filter_channels": model_cfg["filter_channels"],
        "n_heads": model_cfg["n_heads"],
        "n_layers": model_cfg["n_layers"],
        "kernel_size": model_cfg["kernel_size"],
        "p_dropout": model_cfg["p_dropout"],
        "window_size": getattr(model_cfg, "window_size", 4) 
    }

    # 构建低秩Encoder并替换
    low_rank_encoder = LowRankEncoder(**encoder_config).to(device)
    net_g.enc_p.encoder = low_rank_encoder
    logger.info(f"[Rank {rank}] ✅ 低秩Encoder替换完成，类型：{type(net_g.enc_p.encoder)}")

    # ===================== 3. 加载低秩权重（结构匹配） =====================
    if not os.path.exists(LOW_RANK_CKPT_PATH):
        raise FileNotFoundError(f"低秩权重文件不存在：{LOW_RANK_CKPT_PATH}")
    
    try:
        logger.info(f"[Rank {rank}] 加载低秩权重：{LOW_RANK_CKPT_PATH}")
        low_rank_ckpt = torch.load(
            LOW_RANK_CKPT_PATH,
            map_location=device,
            weights_only=True
        )
        net_g_state_dict = low_rank_ckpt.get("net_g", low_rank_ckpt)
        
        # 加载权重（strict=False 忽略解码器等不匹配参数）
        net_g.load_state_dict(net_g_state_dict, strict=False)
        logger.info(f"[Rank {rank}] ✅ 低秩权重加载成功")
    except Exception as e:
        logger.error(f"[Rank {rank}] 权重加载失败：{str(e)}")
        raise

    # ===================== 4. 验证低秩参数加载结果（关键检查） =====================
    if rank == 0:
        logger.info("\n[Rank 0] ===== 验证低秩参数加载结果 =====")
        low_rank_params = []
        for name, param in net_g.named_parameters():
            if "W1" in name or "W2" in name:
                low_rank_params.append((name, param.shape))
                logger.info(f"✅ 加载参数：{name} | 形状：{param.shape}")
        
        # 数量校验
        param_count = len(low_rank_params)
        logger.info(f"\n[Rank 0] 总计加载低秩参数：{param_count} 个（预期：{EXPECTED_LOW_RANK_PARAMS} 个）")
        if param_count != EXPECTED_LOW_RANK_PARAMS:
            raise ValueError(
                f"低秩参数数量不匹配！加载{param_count}个，预期{EXPECTED_LOW_RANK_PARAMS}个\n"
                "请检查：1.权重文件是否正确 2.低秩Encoder替换是否成功"
            )
        logger.info("[Rank 0] =========================")

    # ===================== 5. 参数冻结策略（精准匹配低秩层） =====================
    if FREEZE_NON_LOW_RANK:
        logger.info(f"[Rank {rank}] 执行参数冻结策略...")
        frozen_params = 0
        trainable_params = 0

        # 精准匹配低秩层参数（与权重迁移的参数名完全对齐）
        for name, param in net_g.named_parameters():
            # 匹配规则：enc_p.encoder.attn_layers + conv_q/k/v + W1/W2
            if (
                "enc_p.encoder.attn_layers" in name
                and ("conv_q" in name or "conv_k" in name or "conv_v" in name)
                and ("W1" in name or "W2" in name)
            ):
                param.requires_grad = True
                trainable_params += param.numel()
                if rank == 0:
                    logger.info(f"[Rank 0] 可训练参数：{name} | 参数量：{param.numel():,}")
            else:
                param.requires_grad = False
                frozen_params += param.numel()

        # 统计输出
        if rank == 0:
            logger.info(f"\n[Rank 0] ✅ 参数冻结完成：")
            logger.info(f"  可训练参数：{trainable_params/1e6:.4f} M")
            logger.info(f"  冻结参数：{frozen_params/1e6:.4f} M")
            if trainable_params == 0:
                raise ValueError("无可用的可训练参数！请检查参数匹配规则")

    # 额外冻结解码器（确保不训练）
    for p in net_g.dec.parameters():
        p.requires_grad = False
    if rank == 0:
        logger.info("[Rank 0] ✅ 解码器已强制冻结")

    # ===================== 6. 优化器初始化 =====================
    # 获取可训练参数列表
    trainable_params_list = list(filter(lambda p: p.requires_grad, net_g.parameters()))
    
    if rank == 0:
        logger.info(f"\n[Rank 0] 初始化优化器，可训练参数数量：{len(trainable_params_list)}")
    
    # 空参数检查
    if len(trainable_params_list) == 0:
        raise ValueError(f"[Rank {rank}] 优化器无可用参数！")

    # 初始化AdamW优化器
    optim_g = torch.optim.AdamW(
        trainable_params_list,
        lr=TRAIN_LR,
        betas=hps.train.betas if hasattr(hps.train, 'betas') else (0.8, 0.99),
        eps=hps.train.eps if hasattr(hps.train, 'eps') else 1e-9,
        weight_decay=hps.train.weight_decay if hasattr(hps.train, 'weight_decay') else 0.0
    )
    logger.info(f"[Rank {rank}] ✅ AdamW优化器初始化完成（学习率：{TRAIN_LR}）")

    # ===================== 7. 分布式模型包装 =====================
    if n_gpus > 1:
        net_g = DDP(net_g, device_ids=[rank])
        logger.info(f"[Rank {rank}] ✅ 分布式模型包装完成")

    # ===================== 8. 训练循环（替换为你的业务逻辑） =====================
    logger.info(f"\n[Rank {rank}] ✅ 模型初始化全部完成，开始低秩层微调训练！")
    
    try:
        # 训练参数配置
        epochs = hps.train.epochs if hasattr(hps.train, 'epochs') else 100
        eval_interval = hps.train.eval_interval if hasattr(hps.train, 'eval_interval') else 10
        save_dir = hps.model_dir if hasattr(hps, 'model_dir') else "./checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        # 主训练循环
        for epoch in range(1, epochs + 1):
            logger.info(f"\n[Rank {rank}] ===== Epoch {epoch}/{epochs} =====")
            
            # 训练模式
            net_g.train()
            
            # ===================== 替换为你的真实训练逻辑 =====================
            # 以下是示例框架，需替换为实际的数据加载和前向/反向传播
            # 1. 数据加载示例：
            # for batch_idx, batch in enumerate(train_dataloader):
            #     x, x_lengths, y, y_lengths = [b.to(device) for b in batch]
            #     
            #     # 2. 前向传播
            #     y_hat, l_lengths, attn, *_ = net_g(x, x_lengths, y, y_lengths)
            #     
            #     # 3. 损失计算
            #     loss = compute_loss(y_hat, y, y_lengths, l_lengths)
            #     
            #     # 4. 反向传播
            #     optim_g.zero_grad()
            #     loss.backward()
            #     optim_g.step()
            #     
            #     # 5. 日志输出
            #     if rank == 0 and batch_idx % 10 == 0:
            #         logger.info(f"Batch {batch_idx} | Loss: {loss.item():.4f}")
            # =================================================================

            # 权重保存（每eval_interval个epoch保存一次）
            if rank == 0 and epoch % eval_interval == 0:
                save_path = os.path.join(save_dir, f"G_epoch_{epoch}.pth")
                save_dict = {
                    "net_g": net_g.module.state_dict() if n_gpus > 1 else net_g.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "epoch": epoch,
                    "hps": hps,
                    "low_rank_config": encoder_config
                }
                torch.save(save_dict, save_path)
                logger.info(f"[Rank 0] ✅ 权重已保存：{save_path}")

        # 训练完成保存最终权重
        if rank == 0:
            final_save_path = os.path.join(save_dir, "G_final.pth")
            torch.save({
                "net_g": net_g.module.state_dict() if n_gpus > 1 else net_g.state_dict(),
                "optim_g": optim_g.state_dict(),
                "epochs": epochs,
                "train_config": {"lr": TRAIN_LR}
            }, final_save_path)
            logger.info(f"[Rank 0] ✅ 训练完成，最终权重保存：{final_save_path}")

    except KeyboardInterrupt:
        logger.info(f"[Rank {rank}] 训练被手动中断")
        # 中断时保存临时权重
        if rank == 0:
            interrupt_save_path = os.path.join(save_dir, "G_interrupt.pth")
            torch.save({
                "net_g": net_g.module.state_dict() if n_gpus > 1 else net_g.state_dict(),
                "optim_g": optim_g.state_dict()
            }, interrupt_save_path)
            logger.info(f"[Rank 0] ✅ 中断权重已保存：{interrupt_save_path}")
    except Exception as e:
        logger.error(f"[Rank {rank}] 训练异常：{str(e)}", exc_info=True)
        raise

# ===================== 程序入口 =====================
if __name__ == "__main__":
    # 加载配置文件
    logger.info("📌 加载训练配置文件...")
    hps = get_hparams()
    
    # 检查GPU可用性
    n_gpus = torch.cuda.device_count()
    logger.info(f"📌 系统GPU数量：{n_gpus}")
    logger.info(f"📌 低秩权重路径：{LOW_RANK_CKPT_PATH}")
    logger.info(f"📌 模型保存目录：{hps.model_dir if hasattr(hps, 'model_dir') else './checkpoints'}")

    # 启动训练（单/多GPU适配）
    if n_gpus > 1:
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps))
    else:
        run(0, 1, hps)