# Observation 2: local score context and block utility

本实验检验：**在中心块初始得分相近时，周围块得分是否仍能解释该块对正确答案的实际贡献。** 这里把表述限定为“邻域包含额外信息”，不把相关性直接写成因果影响。

实验直接读取 Observation 1 已完成的逐块替换结果，不需要重新加载模型或执行额外 GPU 前向。对于每个未选候选块，定义：

```text
block utility = baseline label NLL - replacement label NLL
```

utility 越大，说明把该候选换入后正确答案 loss 越低。邻域特征是候选中心周围 `7 x 7` 窗口内所有不重复、未越界且因果合法块的初始得分均值，中心块自身被排除。这样不会把中心得分泄漏到邻域特征中。

为控制中心得分，脚本在每次运行内部将候选按邻域均值的中位数分成高、低两组，再按照中心得分最近原则进行一对一匹配。允许的中心得分差默认不超过该运行中心得分标准差的 `0.15`。分组与配对完全不读取 label loss。随后比较每对候选的 utility，并另外将中心得分的线性影响从邻域和 utility 中同时移除，计算残差相关性。

## 运行

你刚才的结果可以直接这样分析：

```bash
python experiments2/run_observation2.py \
  --input output/ruler_observation/vt_32k_seed42/swap \
  --output output/ruler_observation/vt_32k_seed42/observation2
```

如果默认匹配出的 pair 少于 8 对，程序会停止，而不会用少量样本画结论。此时先检查结果，再适度放宽预先定义的 caliper，例如：

```bash
python experiments2/run_observation2.py \
  --input output/ruler_observation/vt_32k_seed42/swap \
  --output output/ruler_observation/vt_32k_seed42/observation2_caliper025 \
  --score-caliper 0.25
```

不要根据哪一个 caliper 得到更显著的结果再挑选论文数字。正文实验应先固定 caliper，再对所有运行使用相同配置。

## 多样本汇总

单个 query row 适合做现象图，不足以支持普遍结论。建议事先固定若干 sample、layer、head，然后把每个完整 `swap` 目录一次性传入：

```bash
python experiments2/run_observation2.py \
  --input \
    output/run_sample0_l8_h4/swap \
    output/run_sample1_l16_h8/swap \
    output/run_sample2_l24_h12/swap \
  --output output/observation2_aggregate
```

匹配和中心得分残差化都在各运行内部完成，不会把不同层或注意力头的原始得分尺度直接混合。

## 输出

- `observation2.png` 和 `observation2.pdf`：左图是中心得分匹配后的成对 utility，右图是控制中心得分后的邻域分数与 utility 残差。点使用较大尺寸，图例带边框。
- `candidate_context.csv`：每个候选的中心得分、邻域均值、替换 loss、utility 和残差。
- `matched_pairs.csv`：所有匹配对以及高邻域减低邻域的 utility 差。
- `summary.json`：配对均值差、bootstrap 区间、胜率、配对 sign-flip 检验、偏相关和残差置换检验。

只有当高邻域候选的平均 utility 更高，并且配对检验与控制中心得分后的残差关系方向一致时，这组数据才支持 Observation 2。如果结果不成立，脚本仍完整保存结果，不筛选有利候选。

单次运行中的候选窗口彼此重叠，因此统计检验仍属于探索性证据。论文里的主要结论应来自多个预先指定的样本、层和头；`summary.json` 也会保留这一限制说明。

CPU 单元测试：

```bash
python -m unittest discover -s experiments2 -p "test_*.py" -v
```
