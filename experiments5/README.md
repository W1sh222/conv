# Experiment 5: learned versus fixed neighborhood kernels

比较四个完整prefill策略：初始分数、正1×1缩放控制、固定竖线＋主对角线7×7核、Qwen step5000已训练7×7核。固定核由代码构造：中心列7个位置与主对角线7个位置取并集，中心重合只计一次，共13个系数为1，其余为0；它不是单独训练得到的核。采用与已有卷积相同的 replicate padding 和 grouped cross-correlation。

```bash
python experiments5/run_neighborhood_ablation.py \
  --data output/analysis_vt_32k_seed20261008/observation.jsonl \
  --sample-indices 0,1,2,3,4 --ratio 0.65 --stride 8 \
  --output output/experiments5/qwen5000_vt32k
```

默认 `--background sparse`：每个策略独立运行所有层/头，用自身实际激活估计并选择mask，再冻结自身mask重复测量。这比只干预单行更接近整体选择策略的评测。Top-k预算相同，但每个策略的激活可以不同。`--layers/heads/query-blocks` 只指定保存诊断分数图的位置，因此本实验只允许各传一个位置。`--background dense` 改为只在目标层/头使用稀疏选择，其他注意力均为dense，报告时必须区分两种设定。

1×1控制使用单位正增益，Top-k排序与初始策略完全相同，预期除数值漂移外NLL一致；这是流程校验，**不是已训练1×1权重的消融实验**。

比较量为teacher-forced答案token平均NLL和相对初始策略的NLL下降，不是完整RULER生成得分。每个输入四个策略均保留结果，不能只挑改善样本。正式结论应在多个未见输入/任务上复现。

```bash
python experiments3/plot_analysis.py --experiment 5 --input output/experiments5/qwen5000_vt32k
```

保存原始loss、数值漂移、逐输入表、输入级bootstrap汇总和PNG/PDF/SVG配对图。依赖、数据准备及权重默认路径见 `experiments3/README.md`。本程序不更新任何模型权重，也不改写原始评测代码。
