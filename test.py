import torch
import json
import os

# 仅导入VITS原始模块（不导入你的低秩attentions.py）
from models import SynthesizerTrn

# ===================== 配置参数（仅需修改你的原始文件路径） =====================
ORIGINAL_CKPT_PATH = "./pretrained_ljs.pth"  # 原始预训练权重路径
CONFIG_PATH = "./configs/ljs_base.json"      # 原始配置文件路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 辅助函数：打印enc_q所有参数（确认可训练参数） =====================
def print_enc_q_all_params(enc_q_module):
    """打印enc_q的所有可训练参数，确认存在可训练参数"""
    params_list = list(enc_q_module.named_parameters())
    if len(params_list) == 0:
        print("❌ enc_q 无任何可训练参数！")
        return []
    print(f"✅ enc_q 包含 {len(params_list)} 个可训练参数：")
    for idx, (param_name, param) in enumerate(params_list):
        print(f"  序号{idx}：{param_name} | 形状：{param.shape} | 可训练：{param.requires_grad}")
    return params_list

# ===================== 加载配置 + 构建原始模型 =====================
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def build_original_model(config, device):
    """构建原始SynthesizerTrn模型，不做任何低秩替换"""
    hps = config
    model_cfg = hps["model"]
    n_vocab = hps["data"].get("n_vocab", 178)

    # 构建原始完整模型
    original_model = SynthesizerTrn(
        n_vocab=n_vocab,
        spec_channels=hps["data"]["filter_length"] // 2 + 1,
        segment_size=hps["train"]["segment_size"] // hps["data"]["hop_length"],
        inter_channels=model_cfg["inter_channels"],
        hidden_channels=model_cfg["hidden_channels"],
        filter_channels=model_cfg["filter_channels"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        kernel_size=model_cfg["kernel_size"],
        p_dropout=model_cfg["p_dropout"],
        resblock=model_cfg["resblock"],
        resblock_kernel_sizes=model_cfg["resblock_kernel_sizes"],
        resblock_dilation_sizes=model_cfg["resblock_dilation_sizes"],
        upsample_rates=model_cfg["upsample_rates"],
        upsample_initial_channel=model_cfg["upsample_initial_channel"],
        upsample_kernel_sizes=model_cfg["upsample_kernel_sizes"],
        n_speakers=model_cfg.get("n_speakers", 0),
        gin_channels=model_cfg.get("gin_channels", 0),
    ).to(device)

    # 加载原始预训练权重（完整加载，不做任何过滤）
    if os.path.exists(ORIGINAL_CKPT_PATH):
        ckpt = torch.load(ORIGINAL_CKPT_PATH, map_location=device, weights_only=True)
        original_model.load_state_dict(ckpt.get("net_g", ckpt), strict=False)
        print("✅ 原始模型权重加载成功！")
    else:
        raise FileNotFoundError(f"❌ 原始权重文件不存在：{ORIGINAL_CKPT_PATH}")

    return original_model

# ===================== 核心：验证原始enc_q任意参数梯度（兼容所有结构） =====================
if __name__ == "__main__":
    try:
        # 1. 加载配置
        print("="*50 + " 加载原始配置 " + "="*50)
        config = load_config(CONFIG_PATH)
        hps = config
        print(f"✅ 配置文件加载成功：{CONFIG_PATH}")

        # 2. 构建原始模型（无低秩替换）
        print("\n" + "="*50 + " 构建原始模型 " + "="*50)
        original_model = build_original_model(config, DEVICE)
        original_model.train()  # 进入训练模式（关键：推理模式下无梯度）
        print("✅ 原始模型构建完成，已进入训练模式")

        # 3. 打印enc_q所有可训练参数（确认参数存在）
        print("\n" + "="*50 + " 打印enc_q所有可训练参数 " + "="*50)
        enc_q_params = print_enc_q_all_params(original_model.enc_q)
        if len(enc_q_params) == 0:
            raise ValueError("❌ enc_q 无任何可训练参数，无法进行梯度验证")

        # 4. 构造符合要求的输入（与VITS源码输入格式一致）
        print("\n" + "="*50 + " 构造测试输入 " + "="*50)
        B = 1  # 批次大小
        T_txt = 50  # 文本序列长度
        T_mel = 200  # 梅尔序列长度（需大于文本长度）
        n_vocab = hps["data"].get("n_vocab", 178)
        spec_channels = hps["data"]["filter_length"] // 2 + 1  # 梅尔维度

        # 文本输入 x: [B, T_txt]
        x = torch.randint(0, n_vocab, (B, T_txt)).to(DEVICE)
        # 文本长度 x_lengths: [B]
        x_lengths = torch.LongTensor([T_txt]).to(DEVICE)
        # 梅尔频谱 y: [B, spec_channels, T_mel]
        y = torch.randn(B, spec_channels, T_mel).to(DEVICE)
        # 梅尔长度 y_lengths: [B]
        y_lengths = torch.LongTensor([T_mel]).to(DEVICE)
        print(f"输入构造完成：")
        print(f"x shape: {x.shape}, x_lengths shape: {x_lengths.shape}")
        print(f"y shape: {y.shape}, y_lengths shape: {y_lengths.shape}")

        # 5. 前向计算（原始模型完整前向）
        print("\n" + "="*50 + " 执行原始模型前向计算 " + "="*50)
        outs = original_model(x, x_lengths, y, y_lengths)
        z, m_p, logs_p, z_mask, y_hat, ids_slice, attn = outs
        print(f"前向计算完成，z shape: {z.shape}")

        # 6. 计算损失 + 反向传播（与你低秩测试的损失一致）
        print("\n" + "="*50 + " 计算损失并反向传播 " + "="*50)
        original_model.zero_grad()  # 清空残留梯度
        loss = z.mean()  # 与低秩测试损失函数完全一致
        loss.backward()  # 反向传播计算梯度
        print(f"损失计算完成，loss值：{loss.item()}")
        print(f"反向传播完成")

        # 7. 验证enc_q第一个可训练参数的梯度（核心：无需依赖QKV层）
        print("\n" + "="*50 + " 验证原始enc_q参数梯度 " + "="*50)
        first_param_name, first_param = enc_q_params[0]  # 取第一个可训练参数
        print(f"验证enc_q第一个参数：{first_param_name} | 形状：{first_param.shape}")

        # 获取梯度并打印状态
        param_grad = first_param.grad
        if param_grad is not None:
            print(f"{first_param_name} 梯度值（绝对值均值）：{param_grad.abs().mean().item()}")
            # 额外验证enc_q其他参数梯度（可选）
            for param_name, param in enc_q_params[1:3]:  # 验证前3个参数
                grad = param.grad
                if grad is not None:
                    print(f"{param_name} 梯度值（绝对值均值）：{grad.abs().mean().item()}")
                else:
                    print(f"{param_name} 梯度：None")
        else:
            print(f"{first_param_name} 梯度：None")
            # 验证所有enc_q参数梯度
            for param_name, param in enc_q_params:
                if param.grad is not None:
                    print(f"{param_name} 梯度：{param.grad.abs().mean().item()}")
                else:
                    print(f"{param_name} 梯度：None")

        # 8. 结果判断（核心结论）
        enc_q_has_grad = any(param.grad is not None for (_, param) in enc_q_params)
        if enc_q_has_grad:
            print("\n✅ 验证结果：原始enc_q存在非None梯度！说明enc_q参与损失计算，低秩替换后梯度为None是替换引入的问题")
        else:
            print("\n❌ 验证结果：原始enc_q所有参数梯度均为None！说明是源码设计问题，与低秩替换无关")

    except Exception as e:
        print(f"\n❌ 验证失败：{e}")
        raise e