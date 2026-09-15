# Block swap experiment using ground-truth label loss

本实验检验 **较低初始得分块是否能获得更低的正确答案 loss**。不以完整注意力输出为监督，不预设一定存在反例。

## 实验定义

- 指定一条样本、一个 layer、一个 **query head** 和一个 query block（默认最后一个 prompt block）。所有索引从 0 开始。
- 从该 query 行的已选块中，按 **初始得分** 找到最低分块。遇到并列，选择 key block 索引最小者。
- 固定所有其他已选块，将该块逐一换成同一行中 **所有未选且因果合法的块**。不采样、不筛掉不利结果，不修改未来 mask 区域。
- 每次均从基准 mask 开始，恰好改变两个布尔元素；每个 query 行的保留数量不变。
- 默认所有层/头使用稀疏 attention。记录基准 prefill 的全部 mask 后，在所有候选试验中固定这些 mask（包括目标层之后的层）。只有目标位置发生替换；隐藏状态、K/V 和输出仍按各次前向自然变化。
- `--background dense` 可做隔离实验：仅目标层目标头使用稀疏 mask，其余层/头为 dense。它不等价于完整稀疏部署。

## 为什么 label loss 没有答案泄漏

先仅输入 prompt，完成稀疏 prefill 并得到第一个答案 token 的预测。然后以 teacher forcing 方式逐 token 输入之前的正确答案，使用正常 dense decode 预测下一个答案 token。

`label_loss = mean(-log p(y_t | prompt, y_<t))`。

prompt 不计入 loss；答案不进入 prefill 得分图；不自动添加 EOS；每次试验重新建立 KV cache，不复用上一次候选的缓存。KV cache 仅用于正常解码，与 KV 压缩方法无关。该指标衡量正确答案的条件概率，不等价于生成后的 exact match。

prompt 和 label 分别分词，再拼接为生成时的序列边界。空格会影响答案分词，应在所有样本中统一格式。例如裸文本 prompt 为 `Answer:` 时，可把 label 写成 ` 42`；聊天模板则按模型原始格式使用。

## 运行环境

使用本项目已有 Linux CUDA 环境：`torch`、`transformers==4.51.0`、`accelerate`、`triton`、`block_sparse_attn`、`numpy`、`matplotlib`。

本实验直接复用：

- `load_transformers_451.forward_eval_451`：Qwen3 Q/K normalization、RoPE、GQA、KV 和 decode。
- `kernels_conv_block_scores_infer_full`：与 Conv 对应的完整反对角线初始估计。
- `Conv._topk_ratio_mask_from_scores`：固定比例 Top-K。
- `Conv.apply_conv2d_block_map`：可选学习后排序。
- `block_sparse_attn_func`：真正的稀疏 attention 输出，块大小固定为 128。

代码只在本次 Python 进程中临时替换 adapter 的调用，不修改项目原有文件。

## 数据

每行一条 JSON，至少提供明确的 prompt 和正确 label：

```json
{"prompt": "<你的真实长上下文和问题>", "label": "<正确答案>"}
```

也支持 `input`/`output`、`prompt`/`answer`，或最后一条 assistant 消息为正确答案的 `messages`。多个参考答案必须先明确选用哪一个，脚本不会默默挑选最有利的答案。messages 自动套模型聊天模板；普通 prompt 只有设置 `--chat-template` 时才套模板。不自动截断输入。

## 默认观察实验：初始得分 Top-K

在仓库根目录运行（模型路径和数据路径替换为实际位置）：

```bash
python experiments/block_label_swap/run_experiment.py \
  --model /path/to/Qwen3-8B \
  --data /path/to/evaluation.jsonl \
  --sample-index 0 \
  --layer 16 --head 8 --query-block last \
  --selector initial --ratio 0.65 --stride 8 \
  --chat-template \
  --output output/block_label_swap/sample0_l16_h8
```

Llama 使用同一脚本，修改模型路径。`layer=16, head=8` 仅是命令示例，不是经过 loss 搜索挑选的推荐结果。

**`--selector initial` 最适合 Observation 1**：按初始得分建立基准集合，因此候选得分应不高于已选最低分。严格相等的分数不计入“更低得分更好”的反例数量。

## 从学习后的 Conv 集合出发

```bash
python experiments/block_label_swap/run_experiment.py \
  --model /path/to/Qwen3-8B \
  --data /path/to/evaluation.jsonl \
  --layer 16 --head 8 --query-block 31 \
  --selector conv --conv-weights /path/to/conv_kernel.pt \
  --ratio 0.65 --stride 8 --chat-template \
  --output output/block_label_swap/conv_sample0_l16_h8_q31
```

该模式按卷积得分建立集合，但仍按**初始得分**选择移除块。未选候选可能具有更高初始得分；结果中只将严格较低得分且降低 loss 的候选计为目标观察的反例。

长上下文实验必须使用与原实验相同的 RoPE 配置。例如原实验确实使用 YaRN 4 时，增加：

```bash
--rope-factor 4 --rope-original-length 32768 --max-position-embeddings 131072
```

每个候选都执行一次完整 prefill 和答案评分，因此耗时约为 `(未选合法块数 + 2) × 单次评分时间`。先用真实较短样本核验。模型不能放进一张卡时可使用默认 `--device-map auto`；不要把一次候选省略当成完整遍历。

## 图片和结果

- `swap_losses.png` / `.pdf`：目标 query 行的所有合法 key blocks。蓝色=其他已选块，橙色=待移除块（标原始 loss），绿色=未选候选（标换入它后的 label loss）。每格也写初始得分 S。块很多时分页，**每页都是同一个 query 行的不同 key 范围**。
- `mask_overview.png` / `.pdf`：完整目标头的 mask，突出被测 query 行；其他行的未选块未测试，不标替换 loss。
- `score_vs_label_loss.png` / `.pdf`：全部候选的初始得分与最终 label loss，橙色点表示原始块/基准结果。
- `trials.csv`：所有替换的 score、loss 和 delta；`delta_loss < 0` 表示替换更好。
- `trials.jsonl`：同样的记录及答案逐 token loss、teacher-forced argmax（不是自由生成结果）。
- `experiment.json`：配置、答案 token、基准 loss、重复基准 loss、反例数量和完整性状态。
- `block_map.npz`、`frozen_masks.pt`：目标初始得分图与用于每次试验的基准 mask。

运行结束重复一次原始基准，检查数值波动。计数阈值为 `max(--loss-tolerance, 5 * 基准重复差值)`。不要把非常小的舍入差异写成论文结论。单个样本的反例说明“不总是更好”，总体规律需要事先确定多个样本/层/头并完整报告，不能只选最佳候选当作全部结果。

单独重新绘图：

```bash
python experiments/block_label_swap/plot_results.py output/block_label_swap/sample0_l16_h8
```

本地核心逻辑验证（只需 numpy）：

```bash
python -m unittest discover -s experiments/block_label_swap -v
```

这些单元测试检查候选合法性、预算保持、独立替换和反例判定，不能代替真实模型前向验证。
