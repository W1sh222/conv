# 观察实验三：7×7 有符号局部模式与 block utility

## 要回答的问题

实验三只检验下面这个结论：

> 在中心 block 自身的初始得分已知后，它周围的 7×7 block 得分模式是否仍包含关于该 block 最终应获效用的信息？

这里不预设正相关。一个邻居可以提高目标 block 的应得分数，也可以降低它；因此实验使用允许正、负权重同时存在的线性卷积核，而不是实验二的邻域均值。这与 `xattn/src/Conv.py` 中可训练卷积核的作用形式一致。

## 数据与目标量

输入是观察实验一产生的完整 block replacement sweep，每个候选 block 都已经实际换入稀疏注意力 mask 并重新计算正确答案 NLL，无需再次运行 Qwen。

目标量定义为：

```text
utility = baseline label NLL - replacement label NLL
```

- `utility > 0`：换入该 block 后正确答案 loss 降低，说明该 block 有正效用。
- `utility < 0`：换入后 loss 上升，说明该 block 有负效用。
- 这比“初始分数高低”更接近一个 block 最终应该获得的任务相关分数。

## 7×7 特征与 Conv.py 的一致性

每个候选位置提取完整 7×7 score patch：中心、上下左右和对角位置均保留。

- 使用与 `apply_conv2d_block_map` 相同的 `replicate` 边界填充；
- 使用 PyTorch `conv2d` 的 cross-correlation 方向，不翻转卷积核；
- 完整模型中的每一个位置均可学习独立的正权重或负权重；
- 主检验把中心 score 放入控制模型，并用其余 48 个位置检验“中心之外”的增量信息。

## 主检验：真正的留出预测

对于每个 layer/head run，比较两个模型：

1. 控制模型：中心 score + key-block 位置的一次项和二次项；
2. 局部模型：控制模型 + 48 个非中心 7×7 score 特征的有符号 ridge 回归。

采用连续 key-block 分组的 5 折交叉验证。每次测试一段连续位置，并从训练集剔除距测试中心 6 个 block 以内的候选。两个 7×7 patch 的中心距离不超过 6 时可能共享输入元素，因此该 purge 防止相邻重叠 patch 同时落入训练集和测试集。

每层先用该层 utility 方差归一化，再计算主效应：

```text
(控制模型 OOF MSE - 7×7 模型 OOF MSE) / utility 方差
```

这等于 7×7 模型相对控制模型增加的 OOF R²。大于 0 表示周围 block 帮助预测未参与拟合的候选效用。使用 utility 方差而不是控制模型 MSE 作分母，可以避免某一层的控制误差恰好接近零时产生极端百分比。普通 MSE 及其相对改善仍保存在逐层表中。若输入含多个 prompt/sample，程序先在各依赖组内平均各层，再对依赖组等权平均，防止层数较多的 prompt 获得更大权重。

## 显著性检验

主检验使用 circular-shift permutation：候选 utility 沿 key-block 顺序整体循环移动，而 score patch 和空间结构保持不变。这比任意打乱更能保留 utility 的局部自相关。

同一个 VT prompt 在不同 layer 上重复使用，因此这些 layer 不是独立样本。程序会识别复用同一 data/sample、query block 和候选网格的 runs，并对它们施加完全相同的循环位移，从而保留跨层依赖。

若所有输入都属于同一个复用样本组，程序会穷举所有非零循环位移，得到 exact p 值；存在多个独立样本组、组合数很大时，才按 `--permutations` 做 Monte Carlo 抽样。

程序仅在以下条件同时满足时输出：

```json
"supports_observation_in_this_dataset": true
```

- 各 run 的平均 OOF MSE 改善为正；
- 同步 circular-shift 单侧检验 `p < 0.05`。

这个判定不要求某一个 offset 单独显著。原因是 7×7 中多个较弱的正、负作用可以联合预测，但任何单一位置都未必通过多重比较。逐 offset 结果仍通过 max-|r| permutation 做 family-wise error 校正并完整保存。

## 补充证据

实验还输出：

- 每层的 OOF 改善、预测相关和 R²；
- 只控制中心 score 的较宽松结果；
- 完整 7×7、竖线+反斜线、同行、同列、两条对角线的空间消融；
- ridge alpha 从 0.001 到 10 的敏感性分析；
- 每个 offset 在移除中心 score 和位置效应后的正/负偏相关；
- max-stat family-wise 校正后的 offset p 值；
- 每层每折拟合出的有符号卷积核、归一化平均核和符号稳定性；
- 所有候选的原始 7×7 patch 和 OOF 预测，便于复查。

## Linux 运行方式

已有 VT 各层的 `swap` 结果时，直接运行：

```bash
python experiments3/run_observation3.py \
  --input 'output/ruler_observation/vt_32k_seed42_layers/layer_*/swap' \
  --output output/ruler_observation/vt_32k_seed42_layers/observation3_signed7x7 \
  --kernel-size 7 \
  --folds 5 \
  --purge-radius 6 \
  --ridge-alpha 0.1 \
  --permutations 2000 \
  --bootstrap-samples 10000 \
  --seed 42
```

也可以使用包装脚本：

```bash
bash experiments3/run_vt_observation3.sh \
  output/ruler_observation/vt_32k_seed42_layers
```

正式汇报建议把 `--permutations` 提高到 10000；先检查流程时可以使用 500。输出目录必须是新目录或空目录，避免混入旧结果。

只分析若干指定层时，可以显式列出：

```bash
python experiments3/run_observation3.py \
  --input \
    output/ruler_observation/vt_32k_seed42_layers/layer_22/swap \
    output/ruler_observation/vt_32k_seed42_layers/layer_23/swap \
    output/ruler_observation/vt_32k_seed42_layers/layer_28/swap \
  --output output/ruler_observation/vt_selected_observation3
```

## 输出文件

- `summary.json`：主结论、p 值、最强正/负位置、消融、alpha 敏感性和输入元数据；
- `observation3.png` / `observation3.pdf`：层级 OOF 改善、留出预测、平均有符号核、偏相关热图；
- `per_run_metrics.csv`：每层的控制模型和 7×7 模型 OOF 指标；
- `oof_predictions.csv`：每个候选的真实 utility 与严格留出预测；
- `candidate_patches.csv`：每个候选的完整 7×7 输入；
- `kernel_weights.csv`：每个 run、每折、每个 offset 的原始和归一化权重；
- `offset_correlations.csv`：逐 offset 偏相关、FWER p 值、平均拟合权重与符号稳定性；
- `ablation_metrics.csv`：不同空间子集的比较；
- `alpha_sensitivity.csv`：正则强度敏感性。

## 结果应怎样表述

若主判定为 `true`，可以说：

> 在这组预先指定的 VT layer/head 数据中，允许正负权重的 7×7 周边 score 模式，在控制中心 score 和位置后，仍显著改善了对 block intervention utility 的空间隔离留出预测；结果支持局部卷积重评分所依赖的核心观察。

不要直接说“已经证明任意任务都成立”。同一 prompt 的多层结果仍只是一个相关的数据组，最终应在多个预先指定的 prompt、head、seed 和任务上复现。
