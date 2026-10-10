# 两个最终卷积权重的多阶段流程备份

入口：`reproduce_final_weights_multistage.py`。此文件内嵌了六份启动脚本的快照与 Qwen 初始训练日志中的参数，不依赖这些 `.sh` 文件日后的修改，但仍使用仓库中的训练器、数据生成器和权重验证器。

| 模型 | 顺序 | 训练阶段 | 使用的阶段输出 |
|---|---:|---|---|
| Llama | 1 | identity 初始化，24K–32K，18,000 步 | raw step 18,000 |
| Llama | 2 | 96K–128K，按 5,000 步的调度训练 | EMA step 4,000 |
| Llama | 3 | 96K–128K，低学习率，6,000 步 | 末尾 EMA |
| Llama | 4 | 112K–128K，每四次更新加入一次 8K–64K replay，2,000 步 | 最终 Llama EMA |
| Qwen | 1 | identity 初始化，48K–64K，16,000 步，stride 16 | EMA step 16,000 |
| Qwen | 2 | 48K–72K / 64K–96K / 96K–128K，6,000 / 8,000 / 9,250 步，stride 8，YaRN 4 | 第三阶段 EMA step 9,250 |
| Qwen | 3 | native RoPE，8K–64K，5,000 步 | 最终 Qwen EMA step 5,000 |

默认仅查看流程，不训练，也不建立输出目录：

```bash
python ft_scripts/reproduce_final_weights_multistage.py --dry-run
```

在服务器仓库根目录、已激活训练环境后执行：

```bash
python ft_scripts/reproduce_final_weights_multistage.py --model llama --execute
python ft_scripts/reproduce_final_weights_multistage.py --model qwen --execute
```

默认输出在 `output/final_weight_multistage_backup/`，不会覆盖原有权重。可以用 `--output-dir`、`--model-root`、`--nolima-root` 指定位置；初始 Qwen 数据可用 `--qwen-initial-data` 指向原始 16,000 条训练 JSONL。

已生成的阶段输出自动跳过，未完成训练优先使用该阶段自己的训练状态恢复；执行计划、原启动脚本的 SHA256、阶段日志与选定输出路径保存到输出目录中。

## 记录边界

- Qwen 初始训练参数来自保存的训练日志，但原始数据生成命令缺失，因此脚本读取原始训练 JSONL，不猜测或重新生成这一阶段的数据。
- 后续阶段采用现存启动脚本的参数和明确的阶段输出选择；缺少完整历史启动记录，不能证明当时没有额外环境变量覆盖。
- Llama 第二阶段保持 5,000 步的训练调度并选择第 4,000 步输出，不能直接把总步数改成 4,000，否则余弦学习率轨迹会变化。
- 从零指卷积权重从 identity 开始，语言模型仍使用冻结的预训练参数；重训不能保证权重、运行耗时或评测分数逐位相同。
- kernel 对比采用共同训练实现下的阶段权重筛选，不代表各阶段超参数或所选训练步数完全相同，也不宣称采用了独立验证集进行选择。

本地已检查流程展开与初始训练参数的接口兼容性，实际训练仍需服务器 CUDA 环境；本次没有启动训练。
