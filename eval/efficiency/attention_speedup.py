try:
    from xattn.src.Xattention import Xattention_prefill
    XATTN_PREFILL = True
except Exception:
    XATTN_PREFILL = False

try:
    from xattn.src.Flexprefill import Flexprefill_prefill
    FLEXPREFILL_PREFILL = True
except Exception:
    FLEXPREFILL_PREFILL = False

try:
    from xattn.src.Minference import Minference_prefill
    MINFERENCE_PREFILL = True
except Exception:
    MINFERENCE_PREFILL = False

try:
    from xattn.src.Fullprefill import Full_prefill
    FULL_PREFILL = True
except Exception:
    FULL_PREFILL = False

try:
    from xattn.src.Conv import Conv_prefill
    CONV_PREFILL = True
except Exception:
    CONV_PREFILL = False


import os
import pickle
import time

import torch
from tqdm import tqdm
from transformers import StaticCache

from eval.efficiency.generate_prompt import generate_prompt
from xattn.src.load_llama import load_fake_model, FastPrefillConfig
from xattn.threshold.llama_threshold import llama_fuse_8, llama_fuse_16


# =========================
# Conv 配置
# =========================
CONV_WEIGHT_PATH = "/inspire/hdd/global_user/gexinmu-253108100065/Repos/fuyicheng_workshop/Innovator-lm-evaluation-hardness/x-attention-main/xattn/conv_weights/conv_kernel_7x7_ruler_mix_guarded_t06_top16_1e4_step9500_topk08.pt"

# 如果 Conv 仍然 illegal memory access，可以先改成 True 跑通
CONV_SAFE_TOPK = False

# 如果你的 Conv.py 里 triton 路径不稳定，可以改成 False
CONV_USE_TRITON = True

CONV_FALLBACK_TOPK = 8

# True 时额外单独跑一次 XAttention/Conv，调用 return_density=True 并打印平均 density。
# 注意：density 计算会 .item() 同步 GPU，所以不放进 timing benchmark 里。
REPORT_DENSITY = True

# Top-k ratio mode.
# XAttention and Conv use the same per-query causal-visible block ratio.
# For each query block i:
#     K_i = ceil(TOPK_RATIO * visible_key_blocks_i)
# Threshold is still passed for API compatibility, but is ignored when
# topk_ratio is not None.
TOPK_RATIO = float(os.environ.get("TOPK_RATIO", "0.65"))
if not (0.0 < TOPK_RATIO <= 1.0):
    raise ValueError(
        f"TOPK_RATIO must be in (0, 1], got {TOPK_RATIO}"
    )


def benchmark_cuda(fn, num_iterations=50):
    total_time = 0.0

    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.time()

        out = fn()

        torch.cuda.synchronize()
        total_time += time.time() - start_time

        del out

    return total_time / num_iterations


def run_density_once(fn):
    torch.cuda.synchronize()
    out = fn()
    torch.cuda.synchronize()

    if isinstance(out, tuple):
        attn_output, density = out
        del attn_output
        return float(density)

    del out
    return float("nan")


if __name__ == "__main__":

    lens = [4, 8, 16, 32, 64, 128]

    speedups_flex = []
    speedups_xattn_8 = []
    speedups_xattn_16 = []
    speedups_conv_8 = []
    speedups_conv_16 = []
    speedups_minfer = []

    past_key_values = None

    for seq_k in lens:
        print(f"Testing {seq_k}K")
        print(
            f"[same-topk-ratio mode] topk_ratio={TOPK_RATIO:.4f}"
        )

        seq_len = seq_k * 1024

        query_path = f"output/query_{seq_len}.pkl"
        key_path = f"output/key_{seq_len}.pkl"

        config = FastPrefillConfig(metric="xattn", stride=16)
        layer_to_save = 12

        if not os.path.exists(query_path) or not os.path.exists(key_path):
            model, tokenizer = load_fake_model(
                name_or_path="/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Llama-3.1-8B-Instruct",
                layer_to_save=layer_to_save,
                target_len=seq_len,
            )

            input_ids = generate_prompt(tokenizer, seq_len)
            chunk_size = 4096

            if past_key_values is not None:
                past_key_values.reset()
            else:
                past_key_values = StaticCache(
                    config=model.config,
                    batch_size=1,
                    max_cache_len=300000,
                    device=model.device,
                    dtype=model.dtype,
                )

            with torch.no_grad():
                for i in tqdm(
                    range(0, input_ids.size(1), chunk_size),
                    desc="Prefilling",
                    unit="chunk",
                ):
                    chunk = input_ids[:, i: i + chunk_size]

                    output = model(
                        input_ids=chunk,
                        past_key_values=past_key_values,
                        use_cache=True,
                        num_logits_to_keep=1,
                    )

                    past_key_values = output.past_key_values

        with open(query_path, "rb") as f:
            q = pickle.load(f)

        with open(key_path, "rb") as f:
            k = pickle.load(f)

        assert q.shape[-2] == seq_len
        assert k.shape[-2] == seq_len

        if not q.is_cuda:
            q = q.to("cuda")
        if not k.is_cuda:
            k = k.to("cuda")

        q = q.contiguous()
        k = k.contiguous()

        torch.manual_seed(0)

        # FlexPrefill args
        gamma = 0.95
        tau = 0.1

        # XAttention / Conv threshold (ignored when topk_ratio is set)
        threshold_8 = torch.tensor(
            llama_fuse_8,
            dtype=torch.float32,
            device=q.device,
        )[layer_to_save]

        threshold_16 = torch.tensor(
            llama_fuse_16,
            dtype=torch.float32,
            device=q.device,
        )[layer_to_save]

        v = torch.randn(
            q.shape,
            dtype=torch.bfloat16,
            device="cuda",
        ).contiguous()

        num_iterations = 50
        num_warmups = 30

        # =========================
        # Warmup
        # =========================
        for _ in range(num_warmups):

            if XATTN_PREFILL:
                try:
                    Xattention_prefill(
                        q,
                        k,
                        v,
                        stride=8,
                        threshold=threshold_8,
                        use_triton=True,
                        chunk_size=min(32768, seq_len),
                        topk_ratio=TOPK_RATIO,
                    )

                    Xattention_prefill(
                        q,
                        k,
                        v,
                        stride=16,
                        threshold=threshold_16,
                        use_triton=True,
                        chunk_size=min(32768, seq_len),
                        topk_ratio=TOPK_RATIO,
                    )

                except Exception as e:
                    print(f"[WARN] Xattention_prefill warmup failed: {repr(e)}")
                    XATTN_PREFILL = False

            if CONV_PREFILL:
                try:
                    Conv_prefill(
                        q,
                        k,
                        v,
                        stride=8,
                        threshold=threshold_8,
                        use_triton=CONV_USE_TRITON,
                        conv_weight_path=CONV_WEIGHT_PATH,
                        layer_idx=layer_to_save,
                        conv_safe_topk=CONV_SAFE_TOPK,
                        fallback_topk=CONV_FALLBACK_TOPK,
                        chunk_size=min(32768, seq_len),
                        topk_ratio=TOPK_RATIO,
                    )

                    Conv_prefill(
                        q,
                        k,
                        v,
                        stride=16,
                        threshold=threshold_16,
                        use_triton=CONV_USE_TRITON,
                        conv_weight_path=CONV_WEIGHT_PATH,
                        layer_idx=layer_to_save,
                        conv_safe_topk=CONV_SAFE_TOPK,
                        fallback_topk=CONV_FALLBACK_TOPK,
                        chunk_size=min(32768, seq_len),
                        topk_ratio=TOPK_RATIO,
                    )

                except Exception as e:
                    print(f"[WARN] Conv_prefill warmup failed: {repr(e)}")
                    CONV_PREFILL = False

            if FULL_PREFILL:
                try:
                    Full_prefill(q, k, v, causal=False)
                except Exception as e:
                    print(f"[WARN] Full_prefill warmup failed: {repr(e)}")
                    FULL_PREFILL = False

            if FLEXPREFILL_PREFILL:
                try:
                    Flexprefill_prefill(
                        q.transpose(1, 2),
                        k.transpose(1, 2),
                        v.transpose(1, 2),
                        gamma,
                        tau,
                    )
                except Exception as e:
                    print(f"[WARN] Flexprefill_prefill warmup failed: {repr(e)}")
                    FLEXPREFILL_PREFILL = False

            if MINFERENCE_PREFILL:
                try:
                    Minference_prefill(k, q, v)
                except Exception as e:
                    print(f"[WARN] Minference_prefill warmup failed: {repr(e)}")
                    MINFERENCE_PREFILL = False

        # =========================
        # Efficiency Evaluation
        # =========================

        if FLEXPREFILL_PREFILL:
            avg_time_flex = benchmark_cuda(
                lambda: Flexprefill_prefill(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    gamma,
                    tau,
                    topk_ratio=TOPK_RATIO,
                ),
                num_iterations=num_iterations,
            )
        else:
            avg_time_flex = float("nan")

        if XATTN_PREFILL:
            avg_time_xattn_8 = benchmark_cuda(
                lambda: Xattention_prefill(
                    q,
                    k,
                    v,
                    stride=8,
                    threshold=threshold_8,
                    use_triton=True,
                    chunk_size=min(32768, seq_len),
                    topk_ratio=TOPK_RATIO,
                ),
                num_iterations=num_iterations,
            )

            avg_time_xattn_16 = benchmark_cuda(
                lambda: Xattention_prefill(
                    q,
                    k,
                    v,
                    stride=16,
                    threshold=threshold_16,
                    use_triton=True,
                    chunk_size=min(32768, seq_len),
                    topk_ratio=TOPK_RATIO,
                ),
                num_iterations=num_iterations,
            )
        else:
            avg_time_xattn_8 = float("nan")
            avg_time_xattn_16 = float("nan")

        if CONV_PREFILL:
            avg_time_conv_8 = benchmark_cuda(
                lambda: Conv_prefill(
                    q,
                    k,
                    v,
                    stride=8,
                    threshold=threshold_8,
                    use_triton=CONV_USE_TRITON,
                    conv_weight_path=CONV_WEIGHT_PATH,
                    layer_idx=layer_to_save,
                    conv_safe_topk=CONV_SAFE_TOPK,
                    fallback_topk=CONV_FALLBACK_TOPK,
                    chunk_size=min(32768, seq_len),
                    topk_ratio=TOPK_RATIO,
                ),
                num_iterations=num_iterations,
            )

            avg_time_conv_16 = benchmark_cuda(
                lambda: Conv_prefill(
                    q,
                    k,
                    v,
                    stride=16,
                    threshold=threshold_16,
                    use_triton=CONV_USE_TRITON,
                    conv_weight_path=CONV_WEIGHT_PATH,
                    layer_idx=layer_to_save,
                    conv_safe_topk=CONV_SAFE_TOPK,
                    fallback_topk=CONV_FALLBACK_TOPK,
                    chunk_size=min(32768, seq_len),
                    topk_ratio=TOPK_RATIO,
                ),
                num_iterations=num_iterations,
            )
        else:
            avg_time_conv_8 = float("nan")
            avg_time_conv_16 = float("nan")

        # =========================
        # Optional density report
        # =========================
        if REPORT_DENSITY and XATTN_PREFILL:
            density_xattn_8 = run_density_once(
                lambda: Xattention_prefill(
                    q,
                    k,
                    v,
                    stride=8,
                    threshold=threshold_8,
                    use_triton=True,
                    chunk_size=min(32768, seq_len),
                    topk_ratio=TOPK_RATIO,
                    return_density=True,
                )
            )
            density_xattn_16 = run_density_once(
                lambda: Xattention_prefill(
                    q,
                    k,
                    v,
                    stride=16,
                    threshold=threshold_16,
                    use_triton=True,
                    chunk_size=min(32768, seq_len),
                    topk_ratio=TOPK_RATIO,
                    return_density=True,
                )
            )
        else:
            density_xattn_8 = float("nan")
            density_xattn_16 = float("nan")

        if REPORT_DENSITY and CONV_PREFILL:
            density_conv_8 = run_density_once(
                lambda: Conv_prefill(
                    q,
                    k,
                    v,
                    stride=8,
                    threshold=threshold_8,
                    use_triton=CONV_USE_TRITON,
                    conv_weight_path=CONV_WEIGHT_PATH,
                    layer_idx=layer_to_save,
                    conv_safe_topk=CONV_SAFE_TOPK,
                    fallback_topk=CONV_FALLBACK_TOPK,
                    chunk_size=min(32768, seq_len),
                    topk_ratio=TOPK_RATIO,
                    return_density=True,
                )
            )
            density_conv_16 = run_density_once(
                lambda: Conv_prefill(
                    q,
                    k,
                    v,
                    stride=16,
                    threshold=threshold_16,
                    use_triton=CONV_USE_TRITON,
                    conv_weight_path=CONV_WEIGHT_PATH,
                    layer_idx=layer_to_save,
                    conv_safe_topk=CONV_SAFE_TOPK,
                    fallback_topk=CONV_FALLBACK_TOPK,
                    chunk_size=min(32768, seq_len),
                    topk_ratio=TOPK_RATIO,
                    return_density=True,
                )
            )
        else:
            density_conv_8 = float("nan")
            density_conv_16 = float("nan")

        if MINFERENCE_PREFILL:
            avg_time_minfer = benchmark_cuda(
                lambda: Minference_prefill(
                    q,
                    k,
                    v,
                ),
                num_iterations=num_iterations,
            )
        else:
            avg_time_minfer = float("nan")

        if FULL_PREFILL:
            avg_time_flashinfer = benchmark_cuda(
                lambda: Full_prefill(q, k, v, causal=False),
                num_iterations=num_iterations,
            )
        else:
            raise RuntimeError("Full_prefill is unavailable, cannot compute speedup.")

        # =========================
        # Calculate speedups
        # =========================
        speedup_flex = avg_time_flashinfer / avg_time_flex
        speedup_xattn_8 = avg_time_flashinfer / avg_time_xattn_8
        speedup_xattn_16 = avg_time_flashinfer / avg_time_xattn_16
        speedup_conv_8 = avg_time_flashinfer / avg_time_conv_8
        speedup_conv_16 = avg_time_flashinfer / avg_time_conv_16
        speedup_minfer = avg_time_flashinfer / avg_time_minfer

        speedups_flex.append(speedup_flex)
        speedups_xattn_8.append(speedup_xattn_8)
        speedups_xattn_16.append(speedup_xattn_16)
        speedups_conv_8.append(speedup_conv_8)
        speedups_conv_16.append(speedup_conv_16)
        speedups_minfer.append(speedup_minfer)

        print(
            f"{seq_k}K "
            f"full={avg_time_flashinfer:.4f} "
            f"minfer={avg_time_minfer:.4f} "
            f"flex={avg_time_flex:.4f} "
            f"xattn_8={avg_time_xattn_8:.4f} "
            f"xattn_16={avg_time_xattn_16:.4f} "
            f"conv_8={avg_time_conv_8:.4f} "
            f"conv_16={avg_time_conv_16:.4f}"
        )

        print(
            f"{seq_k}K speedup "
            f"minfer={speedup_minfer:.2f} "
            f"flex={speedup_flex:.2f} "
            f"xattn_8={speedup_xattn_8:.2f} "
            f"xattn_16={speedup_xattn_16:.2f} "
            f"conv_8={speedup_conv_8:.2f} "
            f"conv_16={speedup_conv_16:.2f}"
        )

        if REPORT_DENSITY:
            print(
                f"{seq_k}K density "
                f"xattn_8={density_xattn_8:.4f} "
                f"xattn_16={density_xattn_16:.4f} "
                f"conv_8={density_conv_8:.4f} "
                f"conv_16={density_conv_16:.4f}"
            )

    # =========================
    # Output results
    # =========================
    print(
        f"\n{'Length':<10}"
        f"{'Flex':<12}"
        f"{'Xattn 8':<12}"
        f"{'Xattn 16':<12}"
        f"{'Conv 8':<12}"
        f"{'Conv 16':<12}"
        f"{'Minfer':<12}"
    )

    for (
        seq_k,
        speedup_flex,
        speedup_xattn_8,
        speedup_xattn_16,
        speedup_conv_8,
        speedup_conv_16,
        speedup_minfer,
    ) in zip(
        lens,
        speedups_flex,
        speedups_xattn_8,
        speedups_xattn_16,
        speedups_conv_8,
        speedups_conv_16,
        speedups_minfer,
    ):
        print(
            f"{str(seq_k) + 'K':<10}"
            f"{speedup_flex:<12.2f}"
            f"{speedup_xattn_8:<12.2f}"
            f"{speedup_xattn_16:<12.2f}"
            f"{speedup_conv_8:<12.2f}"
            f"{speedup_conv_16:<12.2f}"
            f"{speedup_minfer:<12.2f}"
        )
