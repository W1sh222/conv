# RULER VT → block replacement observation

在仓库根目录、已有的 Linux CUDA 评测环境中运行：

```bash
python experiments/run_ruler_observation.py
```

默认从 `eval/LongBench/config/model2path.json` 读取你的 Qwen3-8B 路径。也可指定：

```bash
python experiments/run_ruler_observation.py \
  --model /path/to/Qwen3-8B \
  --seq-length 32768 --seed 42 \
  --layer 16 --head 8 --query-block last \
  --output output/ruler_observation/vt_32k_seed42
```

输出目录必须为空或不存在。未指定时自动生成带时间戳的目录。默认生成一条样本，并运行已有 `block_label_swap/run_experiment.py`，结束后生成 loss 图。不需要额外传数据文件。

## 数据和实验定义

- 任务是标准 **RULER VT（variable tracking）**，从 `eval/RULER/scripts/synthetic.yaml` 读取参数：一条变量链、四跳赋值关系，共需回答五个变量名。32K 长上下文中的链式追踪用于构造较有挑战性的样本；不宣称它是实测最难任务。
- 直接调用仓库 `variable_tracking.py`，保留其噪声、few-shot 示例和长度控制。模板及 answer prefix 来自当前评测的 `template.py` 和 `synthetic/constants.py`；Qwen3 使用 `/no_think`，Llama 使用 `meta-llama3`。不再套第二次聊天模板。
- 直接调用任务生成器，可以避免 `prepare.py` 的无关 NLTK 下载、覆盖随机种子及 shell 拼接；请求的 seed 会实际传给生成器。生成器的非零退出码会使流水线失败。
- `outputs` 中的五个变量**全部**组成 label，保持生成器顺序，以逗号和空格分隔；不从中挑选一个答案。使用固定序列的 teacher-forced 平均 token NLL，不是 RULER 自由生成后的字符串匹配准确率。其他正确的变量排列可能有不同 NLL。
- 默认测试第 0 个样本、第 16 层、第 8 个 query head、最后一个 prompt query block；这些位置事先指定，未经 loss 搜索挑选。`--num-samples N` 生成 N 条，`--sample-index I` 只评测指定的一条。
- 默认初始得分 Top-K、保留比例 0.65、stride 8、block size 128，所有层/头采用基准稀疏 mask。移除被测行已选块中初始得分最低的块，分别换入该行所有未选且因果合法的块。每次恢复基准，其他层/头/行的 mask 固定。
- 每个候选重新进行 prompt prefill 和正确答案评分，不跳过不利结果，末尾重跑基准检查数值波动。是否有较低得分候选降低 loss，由真实结果决定。

运行前检查模型目录、CUDA、Transformers 版本和依赖。生成后用观察实验同样的分词方式检查实际 prompt+label 长度；超限报错，绝不静默截断。32K 约有 256 个 key blocks，0.65 比例下最后一行约需测试 89 个候选，另加两次基准前向；实际数量取决于输入长度和选择器。

## 其他用法

更长上下文可以沿用你现有 Qwen3 长上下文评测的 YaRN 配置：

```bash
python experiments/run_ruler_observation.py \
  --model /path/to/Qwen3-8B --seq-length 65536 \
  --rope-factor 4 --rope-original-length 32768 \
  --max-position-embeddings 131072
```

Llama 只需改 `--model`，脚本根据 `config.json` 选择对应模板。使用学习后的选择结果时增加 `--selector conv --conv-weights /path/to/weights.pt`；Observation 1 默认仍建议 `initial`。

`--dry-run` 仅打印计划和环境缺失项，不生成数据、不加载模型、不产生实验结果。`--prepare-only` 用真实模型 tokenizer 生成并验证数据，不加载模型权重；此模式不要求 CUDA 和稀疏算子。

依赖沿用现有评测环境：`torch`（CUDA）、`transformers==4.51.0`、`accelerate`、`triton`、`block_sparse_attn`、`numpy`、`matplotlib`，以及生成器的 `PyYAML`、`tqdm`、`tenacity`。不自动安装或修改环境。

## 输出

- `pipeline.json`：配置、任务参数、命令、源文件哈希、长度、状态和结果摘要。`environment_blocked` 表示环境检查失败，并未完成真实实验。
- `raw/vt/validation.jsonl`：原始 RULER 数据。
- `observation.jsonl`：保留原 prompt 并加入完整 label 的实验输入。
- `generation.log` / `observation.log`：生成与模型实验日志。
- `swap/swap_losses.png`、`mask_overview.png`、`score_vs_label_loss.png`：逐块 loss、mask 以及得分与 loss 的关系图，同时输出 PDF。
- `swap/trials.csv`、`trials.jsonl`、`experiment.json`：所有替换结果、基准及重复基准 loss、数值容差和反例统计。

蓝色为其他已选块，橙色为待移除块及基准 loss，绿色为未选候选及替换后 loss。详见 `experiments/block_label_swap/README.md`。

CPU 流水线契约测试（不代表模型实验通过）：

```bash
python -m unittest discover -s experiments -p test_ruler_observation.py -v
python -m unittest discover -s experiments/block_label_swap -p test_swap_core.py -v
```
