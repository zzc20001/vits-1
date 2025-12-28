import torch
import json
import os
# 适配文件结构：低秩模型（attentions.py）
from attentions import Encoder, MultiHeadAttention, LowRankLinear
from models import TextEncoder, SynthesizerTrn

# ===================== 配置参数（仅需修改这4个路径，其余默认） =====================
ORIGINAL_CKPT_PATH = "./pretrained_ljs.pth"  # 原始预训练权重路径
NEW_INIT_CKPT_PATH = "./init_low_rank_model.pth"  # 迁移后权重保存路径
CONFIG_PATH = "./configs/ljs_base.json"  # 你的配置文件路径
RANK = 32  # 低秩分解的秩，和attentions.py中一致
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 核心函数：卷积权重 -> 低秩线性层权重迁移 =====================
def svd_weight_transfer(conv_weight, in_features, out_features, r):
    """
    通过SVD奇异值分解，将1x1卷积权重映射到低秩线性层
    :param conv_weight: 原卷积层权重，shape [out_channels, in_channels, kernel_size]
    :param in_features: 低秩层输入特征数
    :param out_features: 低秩层输出特征数
    :param r: 低秩分解的秩
    :return: W1_init (in_features, r), W2_init (r, out_features) （修正后维度）
    """
    # 去除卷积核维度（1x1卷积，转为2D权重）
    conv_weight_2d = conv_weight.squeeze(-1)  # [out_channels, in_channels] -> [C, C]（此处C=192）
    assert conv_weight_2d.shape == (out_features, in_features), f"权重形状不匹配，预期({out_features},{in_features})，实际{conv_weight_2d.shape}"

    # 执行SVD奇异值分解
    U, S, V = torch.svd(conv_weight_2d.cpu())
    r = min(r, in_features, out_features)  # 确保秩不超过特征数

    # 取前r个奇异值和向量，构建低秩权重
    U_r = U[:, :r]  # [C, r] -> [192, 32]
    S_r = torch.diag(torch.sqrt(S[:r]))  # [r, r] -> [32, 32]
    V_r = V[:, :r]  # [C, r] -> [192, 32]

    # 修正矩阵乘法顺序：确保维度匹配
    W1_init = (U_r @ S_r).T  # [C, r] -> [in_features, r]（[192,32]）
    W2_init = (S_r @ V_r.T).T  # [r, C] -> [r, out_features]（[32,192]）

    return W1_init.to(DEVICE), W2_init.to(DEVICE)

def transfer_multihead_attention_weights(original_attn, new_attn, r):
    """
    迁移 MultiHeadAttention 中 conv_q/conv_k/conv_v 的权重到低秩线性层
    """
    channels = original_attn.channels
    # 迁移 conv_q 权重
    q_W1, q_W2 = svd_weight_transfer(original_attn.conv_q.weight, channels, channels, r)
    with torch.no_grad():
        new_attn.conv_q.W1.weight.copy_(q_W1.unsqueeze(-1))  # [r,C]->[r,C,1]
        new_attn.conv_q.W2.weight.copy_(q_W2.unsqueeze(-1))  # [C,r]->[C,r,1]
    # if original_attn.conv_q.bias is not None:
    #     new_attn.conv_q.W2.bias.data = original_attn.conv_q.bias.data.to(DEVICE)

    # 迁移 conv_k 权重
    k_W1, k_W2 = svd_weight_transfer(original_attn.conv_k.weight, channels, channels, r)
    with torch.no_grad():
        new_attn.conv_k.W1.weight.copy_(k_W1.unsqueeze(-1))
        new_attn.conv_k.W2.weight.copy_(k_W2.unsqueeze(-1))
    # if original_attn.conv_k.bias is not None:
    #     new_attn.conv_k.W2.bias.data = original_attn.conv_k.bias.data.to(DEVICE)

    # 迁移 conv_v 权重
    v_W1, v_W2 = svd_weight_transfer(original_attn.conv_v.weight, channels, channels, r)
    with torch.no_grad():
        new_attn.conv_v.W1.weight.copy_(v_W1.unsqueeze(-1))
        new_attn.conv_v.W2.weight.copy_(v_W2.unsqueeze(-1))
    # if original_attn.conv_v.bias is not None:
    #     new_attn.conv_v.W2.bias.data = original_attn.conv_v.bias.data.to(DEVICE)

    # 迁移 conv_o 权重
    new_attn.conv_o.weight.data = original_attn.conv_o.weight.data.to(DEVICE)
    if original_attn.conv_o.bias is not None:
        new_attn.conv_o.bias.data = original_attn.conv_o.bias.data.to(DEVICE)

    # 迁移相对位置嵌入权重
    if original_attn.window_size is not None and hasattr(original_attn, 'emb_rel_k'):
        new_attn.emb_rel_k.data = original_attn.emb_rel_k.data.to(DEVICE)
        new_attn.emb_rel_v.data = original_attn.emb_rel_v.data.to(DEVICE)

    return new_attn

def transfer_encoder_weights(original_encoder, new_encoder, r):
    """
    迁移 Encoder 中所有注意力层的权重
    """
    # 逐一层迁移注意力权重
    for i in range(len(original_encoder.attn_layers)):
        original_attn = original_encoder.attn_layers[i]
        new_attn = new_encoder.attn_layers[i]
        new_encoder.attn_layers[i] = transfer_multihead_attention_weights(original_attn, new_attn, r)

    # 迁移归一化层和FFN层权重
    for i in range(len(original_encoder.norm_layers_1)):
        # 迁移 norm_layers_1
        original_norm1 = original_encoder.norm_layers_1[i]
        new_norm1 = new_encoder.norm_layers_1[i]
        if hasattr(original_norm1, 'gamma') and original_norm1.gamma is not None:
            new_norm1.gamma.data = original_norm1.gamma.data.to(DEVICE)
        if hasattr(original_norm1, 'beta') and original_norm1.beta is not None:
            new_norm1.beta.data = original_norm1.beta.data.to(DEVICE)

        # 迁移 norm_layers_2
        original_norm2 = original_encoder.norm_layers_2[i]
        new_norm2 = new_encoder.norm_layers_2[i]
        if hasattr(original_norm2, 'gamma') and original_norm2.gamma is not None:
            new_norm2.gamma.data = original_norm2.gamma.data.to(DEVICE)
        if hasattr(original_norm2, 'beta') and original_norm2.beta is not None:
            new_norm2.beta.data = original_norm2.beta.data.to(DEVICE)

        # 迁移 FFN 层
        new_encoder.ffn_layers[i].conv_1.weight.data = original_encoder.ffn_layers[i].conv_1.weight.data.to(DEVICE)
        new_encoder.ffn_layers[i].conv_2.weight.data = original_encoder.ffn_layers[i].conv_2.weight.data.to(DEVICE)
        if original_encoder.ffn_layers[i].conv_1.bias is not None:
            new_encoder.ffn_layers[i].conv_1.bias.data = original_encoder.ffn_layers[i].conv_1.bias.data.to(DEVICE)
        if original_encoder.ffn_layers[i].conv_2.bias is not None:
            new_encoder.ffn_layers[i].conv_2.bias.data = original_encoder.ffn_layers[i].conv_2.bias.data.to(DEVICE)

    return new_encoder

# ===================== 加载配置 + 构建模型 =====================
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def build_original_and_new_encoder(config, device):
    """
    构建原始Encoder和低秩Encoder
    """
    hps = config
    model_cfg = hps["model"]
    hidden_channels = model_cfg["hidden_channels"]
    filter_channels = model_cfg["filter_channels"]
    n_heads = model_cfg["n_heads"]
    n_layers = model_cfg["n_layers"]
    kernel_size = model_cfg["kernel_size"]
    p_dropout = model_cfg["p_dropout"]
    window_size = model_cfg.get("window_size", 4)

    # 1. 导入原始模型
    import attentions_original as attentions_ori
    original_encoder = attentions_ori.Encoder(
        hidden_channels=hidden_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=kernel_size,
        p_dropout=p_dropout,
        window_size=window_size
    ).to(device)

    # 2. 构建低秩Encoder
    new_encoder = Encoder(
        hidden_channels=hidden_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=kernel_size,
        p_dropout=p_dropout,
        window_size=window_size
    ).to(device)

    return original_encoder, new_encoder

def load_original_model_weights(original_encoder, config, ckpt_path, device):
    """加载原始预训练权重到原始Encoder"""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"原始权重文件不存在：{ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    # 提取TextEncoder中的encoder权重
    encoder_state_dict = {}
    if "net_g" in ckpt:
        if "enc_p" in ckpt["net_g"]:
            enc_p_dict = ckpt["net_g"]["enc_p"]
            for k, v in enc_p_dict.items():
                if k.startswith("encoder."):
                    new_k = k.replace("encoder.", "", 1)
                    encoder_state_dict[new_k] = v
    else:
        encoder_state_dict = ckpt.get("encoder", {})

    original_encoder.load_state_dict(encoder_state_dict, strict=False)
    print("原始Encoder权重加载成功！")
    return original_encoder

# ===================== 主迁移流程（核心修复） =====================
if __name__ == "__main__":
    try:
        # 1. 加载配置
        print("="*50 + " 加载配置文件 " + "="*50)
        config = load_config(CONFIG_PATH)
        print(f"配置文件加载成功：{CONFIG_PATH}")

        # 2. 构建原始Encoder和低秩Encoder
        print("\n" + "="*50 + " 构建原始模型和低秩模型 " + "="*50)
        original_encoder, new_encoder = build_original_and_new_encoder(config, DEVICE)
        print("原始Encoder和低秩Encoder构建成功！")

        # 3. 加载原始预训练权重到原始Encoder
        print("\n" + "="*50 + " 加载原始预训练权重 " + "="*50)
        original_encoder = load_original_model_weights(original_encoder, config, ORIGINAL_CKPT_PATH, DEVICE)

        # 4. 迁移权重（核心步骤）
        print("\n" + "="*50 + " 迁移注意力层权重（SVD分解） " + "="*50)
        new_encoder = transfer_encoder_weights(original_encoder, new_encoder, RANK)
        print("所有注意力层权重迁移完成！")

        # 🔴 修复1：先验证低秩Encoder是否包含W1/W2参数
        print("\n" + "="*50 + " 验证低秩Encoder参数 " + "="*50)
        low_rank_params = []
        for name, param in new_encoder.named_parameters():
            if "W1" in name or "W2" in name:
                low_rank_params.append((name, param.shape))
        if len(low_rank_params) == 0:
            raise ValueError("❌ 低秩Encoder中无W1/W2参数！attentions.py定义错误！")
        else:
            print(f"✅ 低秩Encoder包含 {len(low_rank_params)} 个W1/W2参数：")
            for name, shape in low_rank_params:
                print(f"  {name} | {shape}")

        # 5. 构建完整模型（核心修复：先构建空模型，再替换Encoder，最后加载原始权重）
        print("\n" + "="*50 + " 构建完整低秩模型并保存权重 " + "="*50)
        hps = config
        model_cfg = hps["model"]
        
        # 构建空的SynthesizerTrn模型
        new_model = SynthesizerTrn(
            n_vocab=178,  # 替换为你实际的n_vocab（可从config["data"]["n_vocab"]获取）
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
        ).to(DEVICE)


        # 加载原始完整模型权重（只加载非Encoder部分）
        original_ckpt = torch.load(ORIGINAL_CKPT_PATH, map_location=DEVICE, weights_only=True)
        model_state_dict = original_ckpt.get("net_g", original_ckpt)
        
        # 过滤掉enc_p.encoder的权重，避免覆盖低秩层
        filtered_original_dict = {}
        for k, v in model_state_dict.items():
            if not k.startswith("enc_p.encoder."):
                filtered_original_dict[k] = v
        
        # 加载过滤后的原始权重（只加载非Encoder部分）
        new_model.load_state_dict(filtered_original_dict, strict=False)
        new_model.enc_q.pre = torch.nn.Conv1d(80, 192, 7, padding=3).to(DEVICE)
        # 🔴 修复2：先替换低秩Encoder，再加载原始权重（避免覆盖）
        new_model.enc_p.encoder = new_encoder
        # 🔴 修复3：验证完整模型是否包含W1/W2参数
        print("\n" + "="*50 + " 验证完整模型参数 " + "="*50)
        full_model_low_rank_params = []
        for name, param in new_model.named_parameters():
            if "W1" in name or "W2" in name:
                full_model_low_rank_params.append((name, param.shape))
        if len(full_model_low_rank_params) == 0:
            raise ValueError("❌ 完整模型中无W1/W2参数！替换Encoder失败！")
        else:
            print(f"✅ 完整模型包含 {len(full_model_low_rank_params)} 个W1/W2参数：")
            for name, shape in full_model_low_rank_params:
                print(f"  {name} | {shape}")
        for name, p in new_model.named_parameters():
            if "W1" in name or "W2" in name:
                p.requires_grad_(True)
        # x = torch.randn(2, 50, 192).to(DEVICE)
        # y = new_encoder.attn_layers[0].conv_q(x)
        # print(y.shape) 
        # ---------- 最小可复现：检查低秩层是否能正常出梯度 ----------
        new_model.train()
        B, T_txt = 1, 50
        T_mel = 200          # 随便 > T_txt 即可
        print(new_model.enc_q.pre)
        x = torch.randint(0, 178, (B, T_txt)).to(DEVICE)
        x_lengths = torch.LongTensor([T_txt]).to(DEVICE)
        y = torch.randn(B, 80, T_mel).to(DEVICE)  # 80 维梅尔
        y_lengths = torch.LongTensor([T_mel]).to(DEVICE)


# 2. 再 forward &  backward
        outs = new_model(x, x_lengths, y, y_lengths)
        (z, m_p, logs_p, z_mask, y_hat, ids_slice, attn) = outs
        loss =  z.mean()
        loss.backward()
        
        # 3. 最后测梯度
        pq = new_model.enc_p.encoder.attn_layers[0].conv_q
        print('W1 data_ptr:', pq.W1.weight.data_ptr())   # 真实内存地址
        print('W1 id:', id(pq.W1.weight))       
        print('W1 grad:', pq.W1.weight.grad.abs().mean().item() if pq.W1.weight.grad is not None else 'None')
        # 6. 保存迁移后的权重
        new_ckpt = {
            "net_g": new_model.state_dict(),
            "config": config,
            "low_rank_r": RANK,
            "transfer_method": "SVD奇异值分解",
            "original_ckpt": ORIGINAL_CKPT_PATH
        }
        torch.save(new_ckpt, NEW_INIT_CKPT_PATH)
        print(f"\n✅ 低秩模型初始化权重已保存：{NEW_INIT_CKPT_PATH}")

        # 7. 验证参数量对比
        print("\n" + "="*50 + " 验证权重迁移效果 " + "="*50)
        def count_params(model):
            return sum(p.numel() for p in model.parameters()) / 1e6

        original_encoder_params = count_params(original_encoder)
        new_encoder_params = count_params(new_encoder)
        param_reduction = ((original_encoder_params - new_encoder_params) / original_encoder_params) * 100

        print(f"原始Encoder参数量：{original_encoder_params:.4f} M")
        print(f"低秩Encoder参数量：{new_encoder_params:.4f} M")
        print(f"参数量减少比例：{param_reduction:.2f}%")
        print("="*50 + " 权重迁移全部完成！ " + "="*50)

    except Exception as e:
        print(f"\n❌ 权重迁移失败：{e}")
        raise e