# Experiment 4: blocks added and removed by ConvPrefill

默认 Qwen step5000、Top-k=0.65、stride=8，与实验三复用同一数据。

```bash
python experiments4/run_added_removed.py \
  --data output/analysis_vt_32k_seed20261008/observation.jsonl \
  --sample-indices 0,1,2,3,4 --layers 16 --heads 8 --query-blocks last \
  --ratio 0.65 --stride 8 --output output/experiments4/qwen5000_vt32k
```

从同一次初始激活得到初始分数和卷积分数，选择等预算集合 A/B，保存共同保留、被移除、被加入的块。移除块按初始分数降序、加入块按卷积分数降序配对，配对不查看答案；每对均从原始 A 独立交换，**不是累积交换**。另将目标 query 行完整换为 B，测量整体行的效用。其他层/头/query行保持冻结。

`--all-pairs` 可以穷举所有 removed × added 组合，但成本为平方级。默认一一配对。若集合没有差异，单交换统计记为缺失，并仍测量完整行的重复基线效用。

每次loss是teacher-forced答案token NLL，正效用定义为基线减替换后的NLL；不是RULER生成准确率。图包括选块颜色条、单块交换效用分布、完整行效用。输入级统计和数值漂移检查与实验三相同。

```bash
python experiments3/plot_analysis.py --experiment 4 --input output/experiments4/qwen5000_vt32k
```

公共参数、权重路径、数据生成、依赖与输出见 `experiments3/README.md`。实验3/4验证的是给定基线条件下的块效用，不能直接替代部署后的整模型评测。
