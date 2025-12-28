import os
import time
import torch
import numpy as np
import librosa
from thop import profile, clever_format
import models  
import json
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ===================== 1. 基础配置 =====================
#路径
CONFIG_PATH = "./configs/ljs_base.json"
CKPT_PATH = "./pretrained_ljs.pth"  # 原始模型权重
TEST_TEXT_LENGTHS = [50, 100, 200]  # 测试不同文本长度（覆盖常见场景）
BATCH_SIZES = [1, 4, 8]  # 测试不同批量（评估批量推理性能）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP_TIMES = 10  # 预热次数（避免首次推理慢）
TEST_TIMES = 50    # 测试次数（取平均，减少误差）

# ===================== 2. 工具函数：转换为JSON可序列化类型 =====================
def to_serializable(obj):
    """将PyTorch/numpy类型转换为Python原生类型"""
    if isinstance(obj, torch.Tensor):
        return obj.item()  # 张量转原生数值
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # numpy数组转列表
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)  # numpy浮点转Python浮点
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)  # numpy整数转Python整数
    elif isinstance(obj, float):
        return round(obj, 6)  # 浮点保留6位小数
    elif isinstance(obj, int):
        return obj
    else:
        return str(obj)  # 其他类型转字符串
    
# ===================== 2. 加载模型+配置 =====================
def load_model(config_path, ckpt_path, device):
    """加载模型（兼容简化版VITS）"""
    # 加载配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    model_config = config["model"]
    data_config = config["data"]
    
    # 初始化TextEncoder
    text_encoder = models.TextEncoder(
        n_vocab=data_config.get("vocab_size", 512),
        hidden_channels=model_config["hidden_channels"],
        filter_channels=model_config["filter_channels"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        kernel_size=model_config["kernel_size"],
        p_dropout=model_config["p_dropout"],
        out_channels=model_config["hidden_channels"]
    ).to(device).eval()
    
    # 初始化Generator
    generator = models.Generator(
        initial_channel=model_config["inter_channels"],
        resblock=model_config["resblock"],
        resblock_kernel_sizes=model_config["resblock_kernel_sizes"],
        resblock_dilation_sizes=model_config["resblock_dilation_sizes"],
        upsample_rates=model_config["upsample_rates"],
        upsample_initial_channel=model_config["upsample_initial_channel"],
        upsample_kernel_sizes=model_config["upsample_kernel_sizes"],
        gin_channels=0 
    ).to(device).eval()
    
    # 加载权重（兼容weight_norm新API）
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get("generator", checkpoint)
        # 适配新API的weight_orig
        new_state_dict = {}
        for k, v in state_dict.items():
            if "weight" in k and "weight_orig" not in k and "bias" not in k:
                new_state_dict[k.replace("weight", "weight_orig")] = v
            else:
                new_state_dict[k] = v
        generator.load_state_dict(new_state_dict, strict=False)
        print(f"权重加载成功：{ckpt_path}")
    
    return text_encoder, generator, config

# ===================== 3. 性能评估核心函数 =====================
def count_params(model):
    """统计参数量（M）"""
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1e6

def count_flops(model, input_tensor):
    """统计FLOPs（精准值+单位：B/KB/MB/GB）"""
    # 1. 统计原始FLOPs（真值）
    flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
    # 2. 手动转换单位（避免clever_format取整）
    if flops >= 1e9:
        flops_str = f"{flops / 1e9:.4f}G"  # 十亿次（吉）
    elif flops >= 1e6:
        flops_str = f"{flops / 1e6:.4f}M"  # 百万次（兆）
    elif flops >= 1e3:
        flops_str = f"{flops / 1e3:.4f}K"  # 千次（千）
    else:
        flops_str = f"{flops:.4f}B"         # 次（基本单位）
    # 3. 同时返回原始值和带单位的字符串（可选，方便后续计算）
    return flops_str, flops  # 第一个返回值用于显示/保存，第二个用于后续计算

def measure_memory(model, input_tensor):
    """统计显存占用（MB）- 适配多参数输入"""
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    with torch.no_grad():
        # 核心修复：如果输入是元组/列表，解包后传入
        if isinstance(input_tensor, (tuple, list)):
            model(*input_tensor)
        else:
            model(input_tensor)
    mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    return mem

def measure_infer_speed(model_pipeline, text_length, batch_size, device, warmup=10, test=50):
    """
    测量推理速度：text_encoder + generator 全流程
    返回：平均耗时（ms）、QPS（次/秒）
    """
    text_encoder, generator = model_pipeline
    # 构造测试输入
    text = torch.randint(0, 512, (batch_size, text_length)).to(device)
    text_lengths = torch.tensor([text_length]*batch_size).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            feat = text_encoder(text, text_lengths)[0]
            generator(feat)
    torch.cuda.synchronize()
    # 正式测试
    start_time = time.time()
    with torch.no_grad():
        for _ in range(test):
            feat = text_encoder(text, text_lengths)[0]
            generator(feat)
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    avg_time_per_batch = (total_time / test) * 1000  # 每批次耗时（ms）
    avg_time_per_sample = avg_time_per_batch / batch_size  # 单次耗时（ms）
    qps = (test * batch_size) / total_time  # 每秒处理数
    
    return avg_time_per_sample, qps

def calc_mel_mse(audio1, audio2, sr=22050, n_mels=80):
    """计算梅尔频谱MSE（评估音质相似度）"""
    # 转换为numpy数组（避免张量）
    audio1_np = audio1.cpu().numpy().squeeze() if isinstance(audio1, torch.Tensor) else audio1
    audio2_np = audio2.cpu().numpy().squeeze() if isinstance(audio2, torch.Tensor) else audio2
    
    mel1 = librosa.feature.melspectrogram(y=audio1_np, sr=sr, n_mels=n_mels)
    mel2 = librosa.feature.melspectrogram(y=audio2_np, sr=sr, n_mels=n_mels)
    mse = np.mean((mel1 - mel2) **2)
    return to_serializable(mse)

# ===================== 4. 主评估流程 =====================
if __name__ == "__main__":
    # 1. 加载模型
    text_encoder, generator, config = load_model(CONFIG_PATH, CKPT_PATH, DEVICE)
    model_pipeline = (text_encoder, generator)
    
    # 2. 基础信息统计
    print("="*50 + " 基础信息 " + "="*50)
    te_params = count_params(text_encoder)
    gen_params = count_params(generator)
    total_params = te_params + gen_params
    print(f"TextEncoder参数量：{te_params:.2f} M")
    print(f"Generator参数量：{gen_params:.2f} M")
    print(f"总参数量：{total_params:.2f} M")
    
    # 3. FLOPs统计（用文本长度100的输入）
    print("\n" + "="*50 + " FLOPs统计 " + "="*50)
    test_feat = torch.randn(1, config["model"]["hidden_channels"], 100).to(DEVICE)
    gen_flops = count_flops(generator, test_feat)
    print(f"Generator FLOPs：{gen_flops}")
    
    # 4. 显存统计
    if DEVICE.type == "cuda":
        print("\n" + "="*50 + " 显存统计 " + "="*50)
        te_mem = measure_memory(text_encoder, (torch.randint(0,512,(1,100)).to(DEVICE), torch.tensor([100]).to(DEVICE)))
        gen_mem = measure_memory(generator, test_feat)
        print(f"TextEncoder显存占用：{te_mem:.2f} MB")
        print(f"Generator显存占用：{gen_mem:.2f} MB")
    
    # 5. 推理速度测试（不同文本长度+批量）
    print("\n" + "="*50 + " 推理速度测试 " + "="*50)
    print(f"测试条件：预热{WARMUP_TIMES}次，测试{TEST_TIMES}次，设备={DEVICE}")
    print(f"{'文本长度':<10} {'批量':<5} {'单次耗时(ms)':<15} {'QPS(次/秒)':<10}")
    print("-"*50)
    for text_len in TEST_TEXT_LENGTHS:
        for bs in BATCH_SIZES:
            avg_time, qps = measure_infer_speed(model_pipeline, text_len, bs, DEVICE)
            print(f"{text_len:<10} {bs:<5} {avg_time:.2f} {qps:.2f}")
    
    # 6. 音质基准（可选：对比原始音频和合成音频的梅尔MSE）
    print("\n" + "="*50 + " 音质基准 " + "="*50)
    # 构造固定输入，生成测试音频
    with torch.no_grad():
        text = torch.randint(0, 512, (1, 100)).to(DEVICE)
        text_lengths = torch.tensor([100]).to(DEVICE)
        feat = text_encoder(text, text_lengths)[0]
        audio = generator(feat)
    print(f"合成音频长度：{audio.shape[-1]} 采样点")
    print(f"合成音频梅尔频谱MSE（自对比）：{calc_mel_mse(audio, audio):.6f}")  # 基准MSE
    
    # 保存基准数据到文件
    benchmark_data = {
        "total_params_M": total_params,
        "generator_flops": gen_flops,
        "infer_speed": {
            f"text_len_{tl}_bs_{bs}": {
                "avg_time_ms": measure_infer_speed(model_pipeline, tl, bs, DEVICE)[0],
                "qps": measure_infer_speed(model_pipeline, tl, bs, DEVICE)[1]
            } for tl in TEST_TEXT_LENGTHS for bs in BATCH_SIZES
        },
        "mel_mse": calc_mel_mse(audio, audio)
    }
    with open("vits_benchmark_original.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=4)
    print("\n基准数据已保存到：vits_benchmark_original.json")