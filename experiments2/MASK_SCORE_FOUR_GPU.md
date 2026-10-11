# 四卡全行块替换实验

默认预先指定四组（层和查询头均从0编号），不根据观察结果选择：

| 当前可见卡编号 | Layer | Query head |
| --- | --- | --- |
| 0 | 6 | 2 |
| 1 | 14 | 16 |
| 2 | 22 | 24 |
| 3 | 30 | 4 |

这四组用于观察不同层和头的变化；同一输入上改变层和头不构成四个独立数据样本，也不能单独归因于层或头变化。

## 一次启动

服务器仓库根目录中执行，确保作业分配了四张能单独容纳Qwen3-8B和32K提示的GPU：

```bash
conda activate fyc_qwen
python -u experiments2/run_mask_score_four_gpu.py
```

默认读取原全行结果 `output/ruler_observation/mask_score_layer14_head8/experiment.json`，继承其中模型路径、VT样本路径、sample index、比例0.65、stride=8、初始得分选择、稀疏背景、精度、种子和RoPE等设置。不会重新生成VT样本或加载训练权重。默认选择取决于参考设置；若参考实验用卷积选择，则也会继承，务必核对dry-run中的settings。

可先打印实际命令，不加载模型或写入结果：

```bash
python experiments2/run_mask_score_four_gpu.py --dry-run
```

如果参考文件不在默认位置，或输入和模型移动过：

```bash
python -u experiments2/run_mask_score_four_gpu.py \
  --reference output/ruler_observation/mask_score_layer14_head8/experiment.json \
  --data output/ruler_observation/vt_32k_seed42_q220_layers/_vt_data/observation.jsonl \
  --model /inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Qwen3-8B \
  --gpus 0,1,2,3
```

`--gpus`是当前进程可见卡列表的索引：若调度器已设 `CUDA_VISIBLE_DEVICES=4,5,6,7`，默认0,1,2,3会映射为4,5,6,7。每个子进程只可见自己的卡，模型以 `--device-map cuda:0` 加载，避免四个任务同时跨卡占用。默认四组适合Qwen3-8B，其他模型会在启动前检查层和头是否越界。

## 流程和出图

复用 `run_mask_score.py` 的完整流程：逐行移出最低初始得分的已选块，逐个换入每一个原始未选合法块，每次从基线掩码和空KV重新计算完整答案NLL；不累计替换，保持每行预算和其他掩码不变。`query-block=last`只用于基线记录，实际遍历所有可测试行。重复基线后用 `max(1e-5, 5*drift)` 判断改善。

每组完成后复用 `replot_mask_score.py` 画蓝白橙灰选择图，图例位于右上灰色区，导出PNG、PDF、SVG和布局记录。实际推理失败或未跑完不会画完整结果。

结果根目录：`output/ruler_observation/mask_score_four_gpu/`，包含四个 `layerN_headM/` 实验目录、四份独立日志、`four_gpu_manifest.json` 和 `four_gpu_summary.csv`。汇总包含基线NLL、替换数、改善数、改善比例、容差及重复基线漂移；改善比例的分母是实际测试的未选合法块，不包括未来灰色块。

论文图片：`acl_revision/Figs/mask_score_four_gpu/mask_score_layer6_head2.pdf` 等四组，同名PNG/SVG。原始Layer14/Head8图片不覆盖。

## 接续和故障

中断后原命令重新执行：有元数据但未完成的组自动加 `--resume`，已完成的组只检查并重绘，尚未开始的组正常启动；配置变化会拒绝复用已有结果。输出根目录的 `.launcher.lock` 防止重复启动同一套实验；硬杀进程后如果锁遗留，确认原launcher和四个worker均已结束再手动删除锁文件，不能在旧worker还写入时启动第二套。

不需要同时启动四条独立命令；一张卡不足以加载模型或发生OOM时，请在该组日志中排查，不应改为跨卡自动加载来与其他任务争抢显存。
