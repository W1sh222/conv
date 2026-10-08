# Experiment 3: learned score ranking versus intervention utility

三个实验一起运行（已激活 `fyc_qwen`）：

```bash
bash experiments3/run_all_experiments.sh
```

默认复用 `output/ruler_observation/vt_32k_seed42/observation.jsonl`，结果保存到新建的 `output/experiments3_5/qwen5000_<时间>_<进程号>/`，分别包含 `experiment3/4/5` 和各自日志。三个程序顺序执行，任一失败就停止，不覆盖已有结果。

指定数据和输出目录：

```bash
bash experiments3/run_all_experiments.sh \
  --data output/analysis_vt_32k_seed20261008/observation.jsonl \
  --sample-indices 0,1,2,3,4 \
  --output-root output/experiments3_5/qwen5000_vt32k_new
```

`--dry-run` 只检查三个入口的配置；`--help` 查看选项。额外的公共Python参数会转发给三个实验。

使用 Qwen step5000 的现有 EMA 权重，不重新训练。初始分数来自仓库的反对角线估计器，**不是完整 XAttention 方法的复现**。本实验检验卷积后的分数排序是否更符合最终答案的任务效用，不预设结果。

## 数据准备与运行（Linux / CUDA）

在仓库根目录、`conda activate fyc_qwen` 后运行。可以直接复用已有 `observation.jsonl`。建议另生成未用于训练、且预先确定的多样本数据：

```bash
python experiments/run_ruler_observation.py \
  --model /inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Qwen3-8B \
  --seq-length 32768 --num-samples 5 --seed 20261008 \
  --prepare-only --output output/analysis_vt_32k_seed20261008

python experiments3/run_ranking_utility.py \
  --data output/analysis_vt_32k_seed20261008/observation.jsonl \
  --sample-indices 0,1,2,3,4 --layers 16 --heads 8 --query-blocks last \
  --ratio 0.65 --stride 8 --output output/experiments3/qwen5000_vt32k
```

默认权重为仓库下 `xattn/qwen_weights/conv_qwen3_t065_longbench_stage4_v2/conv_kernel_7x7_qwen3_t065_longbench_replay_native_8k64k_s8_bf16_ema_step5000.pt`。可用 `--conv-weights` 覆盖，模型也可用 `--model` 覆盖。

## 干预与统计

- 原始策略为每个因果 query 行保留 `ceil(0.65 * 可见块数)`；block size=128。
- 在指定层/头/query行，移除初始得分最低的已选块，逐个换入所有合法未选块。其他所有层、头和行的 mask 都冻结为原始初始策略；每个试验重新建立 KV cache。
- 只用 prompt 估计分数；答案仅用于 teacher-forcing 的平均 token NLL（不含 EOS）。`utility = baseline NLL - replacement NLL`，正值表示改善。这不是生成后的 RULER 官方分数。
- 初始与卷积分数从同一套基线激活计算，在同一候选集合上比较 Spearman 相关、Top-5/10/20 候选平均效用、改善比例；候选原始排名按完整未选池保存。
- `--max-candidates 20` 可用于首次检查，按固定种子随机抽取候选，不按答案loss挑选。正式结果默认穷举。多头/多层可用逗号分隔，但不能当作独立输入复制。
- 重复基线测量数值漂移，改善容差取 `max(1e-5, 5 * drift)`。多个独立输入先分别汇总，再 bootstrap，避免把同一输入的不同头/块当成独立样本。

每个 case 保存 `experiment.json`、`trials.jsonl/csv`、`block_maps.npz`；根目录保存 `cases.csv`、`summary.json` 和 `plots/` 下 PNG/PDF/SVG。输出必须是新目录，旧结果不覆盖。

重画：

```bash
python experiments3/plot_analysis.py --experiment 3 --input output/experiments3/qwen5000_vt32k
```

`--dry-run` 检查配置而不加载模型。`--no-plots` 暂停绘图。依赖仓库 CUDA / Triton / block_sparse_attn 和 Transformers 4.51.x 环境。128K 输入须显式提供与正式评测一致的 RoPE 设置；程序不截断超长输入。

本地 CPU 测试：`python experiments3/test_analysis.py`。绘图保持零参考线、使用不同点型；所有点都是真实记录。实际推理结束后需在论文最终尺寸检查 PDF 字体和图例，不能将单输入结果推广为整个基准的结论。
