import os
import time
import torch
import numpy as np
import librosa
from thop import profile
import models
import json
import warnings
warnings.filterwarnings("ignore")
USE_FP16 = False
# ------------------ 基础配置 ------------------    
CONFIG_PATH = "./configs/ljs_base.json"
CKPT_PATH = "./pretrained_ljs.pth"

TEST_TEXT_LENGTHS = [50, 100, 200]
BATCH_SIZES = [1, 4, 8]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP_TIMES = 10
TEST_TIMES = 50

# ------------------ 基础工具 ------------------

def to_serializable(x):
    if isinstance(x, torch.Tensor): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, float): return round(x,6)
    return x

# ------------------ 模型加载 ------------------

def load_model(config_path, ckpt_path, device):
    import json
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    m, d = cfg["model"], cfg["data"]

    text_encoder = models.TextEncoder(
        n_vocab=d.get("vocab_size",512),
        hidden_channels=m["hidden_channels"],
        filter_channels=m["filter_channels"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        kernel_size=m["kernel_size"],
        p_dropout=m["p_dropout"],
        out_channels=m["hidden_channels"]
    ).to(device).eval()

    generator = models.Generator(
        initial_channel=m["inter_channels"],
        resblock=m["resblock"],
        resblock_kernel_sizes=m["resblock_kernel_sizes"],
        resblock_dilation_sizes=m["resblock_dilation_sizes"],
        upsample_rates=m["upsample_rates"],
        upsample_initial_channel=m["upsample_initial_channel"],
        upsample_kernel_sizes=m["upsample_kernel_sizes"],
        gin_channels=0
    ).to(device).eval()

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        sd = ckpt.get("generator", ckpt)
        new_sd = {}
        for k,v in sd.items():
            if "weight" in k and "weight_orig" not in k and "bias" not in k:
                new_sd[k.replace("weight","weight_orig")] = v
            else:
                new_sd[k] = v
        generator.load_state_dict(new_sd, strict=False)
    return text_encoder, generator

# ------------------ 统计函数 ------------------

def count_params(model):
    return sum(p.numel() for p in model.parameters())/1e6

def count_model_flops(te, gen, text_len=100):
    text = torch.randint(0,512,(1,text_len)).to(DEVICE)
    lens = torch.tensor([text_len]).to(DEVICE)
    with torch.no_grad():
        feat = te(text,lens)[0]
    te_f,_ = profile(te, inputs=(text,lens), verbose=False)
    gen_f,_ = profile(gen, inputs=(feat,), verbose=False)
    return te_f, gen_f, te_f+gen_f

def measure_memory_safe(model, inputs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.no_grad():
        model(*inputs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()/1024/1024

def measure_infer_speed(pipeline, tl, bs):
    te, gen = pipeline
    text = torch.randint(0,512,(bs,tl)).to(DEVICE)
    lens = torch.tensor([tl]*bs).to(DEVICE)
    with torch.no_grad():
        for _ in range(WARMUP_TIMES):
            gen(te(text,lens)[0])
    torch.cuda.synchronize()
    st = time.time()
    with torch.no_grad():
        for _ in range(TEST_TIMES):
            gen(te(text,lens)[0])
    torch.cuda.synchronize()
    t = time.time()-st
    avg = (t/TEST_TIMES)*1000/bs
    qps = (TEST_TIMES*bs)/t
    return avg, qps

def mel_mse(a,b):
    a,b = a.cpu().numpy().squeeze(), b.cpu().numpy().squeeze()
    return float(np.mean((librosa.feature.melspectrogram(a)-
                           librosa.feature.melspectrogram(b))**2))

# ------------------ 主程序 ------------------

if __name__ == "__main__":
    te, gen = load_model(CONFIG_PATH, CKPT_PATH, DEVICE)
    if USE_FP16:
        te = te.half()
        gen = gen.half()
    pipeline = (te, gen)

    print("Params(M):", count_params(te), count_params(gen))
    tef, genf, totf = count_model_flops(te, gen)
    print("FLOPs(G):", tef/1e9, genf/1e9, totf/1e9)

    if DEVICE.type=="cuda":
        text = torch.randint(0,512,(1,100)).to(DEVICE)
        lens = torch.tensor([100]).to(DEVICE)
        feat = te(text,lens)[0]
        if USE_FP16:
            feat = feat.half()
        print("Memory(MB):", 
              measure_memory_safe(te,(text,lens)),
              measure_memory_safe(gen,(feat,)))

    print("\n" + "="*90)
    print("Inference Speed Benchmark (Single Sample Latency & Throughput)")
    print(f"Device: {DEVICE} | Warmup={WARMUP_TIMES} | Test loops={TEST_TIMES}")
    print("="*90)
    print(f"{'TextLen':<10}{'Batch':<10}{'Latency(ms/sample)':<25}{'Throughput(QPS)':<20}")
    print("-"*90)

    speed_cache = {}
    for tl in TEST_TEXT_LENGTHS:
        for bs in BATCH_SIZES:
            avg, qps = measure_infer_speed(pipeline, tl, bs)
            speed_cache[f"text{tl}_bs{bs}"] = {
                "latency_ms_per_sample": round(avg,3),
                "qps": round(qps,2)
            }
            print(f"{tl:<10}{bs:<10}{avg:<25.3f}{qps:<20.2f}")

    # 保存baseline
    with open("baseline_metrics.json","w") as f:
        json.dump(speed_cache,f,indent=4)
