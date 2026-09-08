# 08 · 数据集与训练逻辑剖析 —— 为何半监督 mAP 提不上去，以及如何逼近全监督

> 目的：结合对 `D:\CV\custom_muti` **图像特征**的实测统计，和 `train.py` / `mlgcn/` 的**训练逻辑**，
> 解释“全监督 0.78 vs 半监督 0.585”的缺口来源，并给出**按优先级排序、附论文与源码论据**的改进建议。

---

## 1. 数据集图像特征（实测）

### 1.1 基本统计
| 项目 | 数值 | 说明 |
|---|---|---|
| train / val | 2209 / 951 | 6 个 `01~06` 源前缀（多相机） |
| 图像分辨率 | **统一 1082×932** | 文件名含时间戳（`-240523093249-`）→ 监控/工业场景的**视频采样帧** |
| 扩展名 | 全部 `.jpg` | — |
| 标签数/图 | 平均 **3.67**，min 1 / max 10 | **稠密多标签** |
| 类别数 | 15 | — |

> 训练时 `img_size=224`，即把 1082×932 **缩到 224×224（约 5x 下采样）**。这类工业/监控画面里
> 目标往往较小、较远（如 person、rescue equipment、container），下采样会丢失小目标细节。
> 尽管当前模型默认 **MP 全局池化只取最大激活**，对“小目标”仍不友好——这是稀有类 AP 低的结构性原因之一。

### 1.2 类别极不平衡（train 标注重数，最频/最稀 ≈ 20 倍）
```
cable 1575, metal fixture 1501, tube 936, indicator 923,
electronic equipment 603, anchoring equipment 603, mining equipment 532,
rail area 293, electrical equipment 267, tools and materials 237,
support equipment 224, door 134, container 119, rescue equipment 89, person 77
```

### 1.3 不同相机源类别差异大（域偏移 / domain shift）
按前缀统计的每类数量差别显著：

| 源 | cable | metal tube | person | rescue | support | rail | 来源特点 |
|---|---|---|---|---|---|---|---|
| 01 | 329 | 390 / 177 | 12 | 30 | 30 | 40 | 均衡 |
| 04 | 212 | 252 / 161 | **4** | **0** | **0** | **102** | 缺 rescue/support |
| 05 | **24** | 46 / 51 | 3 | **0** | **0** | **0** | 几乎全缺稀有类（仅 80 图） |
| 06 | **430** | 320 / 287 | 26 | 17 | 16 | 94 | cable/高频类为主 |

- 某些类**只出现在部分相机**（rescue/support/rail 在 04/05 源为 0）。
- 含义：单一模型必须跨 6 个相机学**域无关**特征；否则容易把“相机背景/视角”当成类别特征，
  这既限制泛化，也会让“只看该相机”的类别学不好。

### 1.4 半监督可用监督信号极弱（最致命的数字）
按 `by_prefix` + `labeled_ratio=0.2`（`data/custom_dataset.py` 同款逻辑，seed=42）模拟 20% 标注子集（**仅 439 张**）：

```
cable 319, metal fixture 298, indicator 187, tube 185, electronic eq 141,
anchoring eq 124, mining eq 84, electrical eq 58, rail 56, support 51,
tools 46, container 30, door 30, person 22, rescue equipment 19
```

- **person 只有 22 个、rescue equipment 只有 19 个正样本**——监督损失对这些类几乎无信号；
- 无监督伪标签又天然偏向高频类（模型对稀有类置信度低、`util_ratio` 只 ~0.30），
  于是稀有类既无监督信号、又无可靠伪标签 → AP 塌缩到 0.21–0.32。

### 1.5 训练/验证场景重叠（评价偏差提示）
- val 951 图同样来自 6 个相机（`01:227, 02:145, 03:165, 04:122, 05:35, 06:257`），与 train **同源同分布**。
- 这是“同相机域内”评价（对任务合理），但要注意：若 val 帧与 train 帧来自相近时间/场景，
  指标可能偏乐观（模型记忆相机 pattern）；跨相机泛化不会体现在该 val 上。

---

## 2. 现有代码训练逻辑的问题

| # | 位置 | 问题 | 影响 |
|---|---|---|---|
| 1 | `train.py:407` `algo.num_train_iter = epochs * len(lb_loader)` | 迭代数随 `labeled_ratio` 改变（ratio=0.05 时每 epoch 仅 ~13 batch）；LR 余弦 `T_max` 也随之不同 | **ratio 扫描混入“训练步数/调度差异”**，低 ratio 既缺标签又欠训练，结论被污染 |
| 2 | `configs/freematch_mlgcn.yaml:10 backbone: resnet101` 但 `train.log` 显示 `resnet50` | 配置与实际运行 backbone 不一致 | 结论与配置不符，可比性受损 |
| 3 | `mlgcn/losses.py` ASL；`configs:15 loss: asl` | ASL 面向**百万级超长尾**，对小规模稠密失衡数据负收益（BCE C=0.6015 > ASL D=0.5849；全监督 A=0.78 > B=0.74） | 用错损失，反而掉点 |
| 4 | `mlgcn/freematch.py` `_sup_loss()`：`class_weight = 1.0/(label_hist*C+0.1)` 且 `clamp(0.5,5.0)` | ①上限仅 5.0，压不住 20 倍失衡；②`label_hist` 用**无标注预测**`probs>0.5`更新，且训练初期≈均匀 → 类别平衡权重**跟随模型自身长尾偏置**，形成“正反馈” | 无法纠正长尾，甚至放大 |
| 5 | `mlgcn/freematch.py` `train_step()`：伪标签掩码 `probs > thresh` | `util_ratio` 0.767→~0.30（约 70% 伪标签被丢弃）；模型 macro-P 0.42 / macro-R 0.73（**过预测**） | 无监督一致性放大了过预测，伪标签多为“全预测正类” |
| 6 | `mlgcn/freematch.py` `entropy_loss()`：`fairness_weight=0.05`、`class_weight.clamp(max=5.0)`；`lambda_e=0.005` | 熵正则“公平项”权重极低，对 20 倍失衡无约束 | 无法抑制“塌缩到高频类” |
| 7 | `mlgcn/models.py` ML-GCN 图头 + `custom_adj.pkl` | SSL 下 D≈E≈F（0.585/0.590/0.575），图头**零增益** | 预生成邻接相关性弱/与领域不适配，图头成了摆设 |
| 8 | `mlgcn/freematch.py` 伪标签用 soft `sigmoid(logits/T)` | 对稠密多标签，soft target 易“模糊/过预测”，BE 一致性把置信度推向两端 | 伪标签质量不稳 |
| 9 | 阈值 `time_p*p_model/max`，`clamp(0.25,0.90)`、分位 0.9 | 下限 0.25 对稀有类仍偏高；`p_model` 越小越难过阈值 | 稀有类更难利用无标签 |

---

## 3. 如何有效逼近全监督（按优先级，附论文 + 源码）

> 先给现实预期：**全监督上界本身很低**（A 中 person/rescue/tools 也仅 AP 0.51–0.67），
> 因为稀有类在数据里本就少且是小目标。因此重点是把**稀有类**的 mAP 拉起来。

### P0 —— 直击类别不平衡（最关键，也是当前最大瓶颈）
1. **用“有效样本数”类别平衡损失替代 `clamp(0.5,5.0)`**，并在监督损失里用**标注子集的真实类别计数**而非无标注预测。
   - 论文：Class-Balanced Loss Based on Effective Number of Samples（Cui et al., CVPR 2019）
   - 源码：[richardaecn/class-balanced-loss](https://github.com/richardaecn/class-balanced-loss)
   - 修改点：`mlgcn/freematch.py` `_sup_loss()`；`class_weight[c]=(1-β)/(1-β^{n_c})`，β≈0.999。
2. **稀有类过采样 / 类别均衡 batch**（`train.py` `build_loaders()` 用 `WeightedRandomSampler`，权重∝1/类频次）。
   - 源码：同上 `class-balanced-loss` 的 resample 实现；参考 `torch.utils.data.WeightedRandomSampler`。
3. **长尾半监督专用方法**（伪标签类别重平衡）。
   - DASO：Distribution-Aware Semantics-Oriented Pseudo-Label（Oh et al., CVPR 2021），[stablegradients/daso](https://github.com/stablegradients/daso)
   - CReST：Class-Rebalancing Self-Training（Wei et al., CVPR 2021），[google-research/crest](https://relatedrepos.com/gh/google-research/crest)
   > 二者用“分布感知阈值/类别重平衡”缓解“伪标签偏向多数类”，正好对应你的 `util_ratio` 塌缩与长尾问题。

### P1 —— 提升伪标签质量与利用率
4. **降低/放宽阈值**（把 `_per_class_threshold` 下限从 0.25 降到 ~0.15，分位从 0.9 降到 ~0.7，`time_p` 初值 0.3），提高 `util_ratio`。
   - FreeMatch（Wang et al., ICLR 2023）：[microsoft/Semi-supervised-learning](https://github.com/microsoft/Semi-supervised-learning)（你已改自其 FreeMatch）
   - FlexMatch（Zhang et al., NeurIPS 2021）：同库；课程式伪标签。
5. **无标签一致性用置信度软权重**，而不是二值掩码；对稠密多标签可**只保留每图 top-k 正类**作伪标签，避免过预测。
   - UDA（Xie et al., ECCV 2020）：[google-research/uda](https://github.com/google-research/uda)
   - Mean Teacher（Tarvainen & Valpola, NeurIPS 2017）：[CuriousAI/mean-teacher](https://github.com/CuriousAI/mean-teacher)
6. **增强熵正则公平项**：`lambda_e` 从 0.005 → 0.02~0.05，`fairness_weight` 0.05→0.1~0.2，`class_weight` 上限去掉/提到 20+。

### P2 —— 模型与特征（解决“图头零增益”+“小目标”）
7. **更强骨干 + 多尺度**（换 ConvNeXt/Swin；或把 `img_size` 提到 384/448，`configs/..._384.yaml` 已备好）。
   - ConvNeXt（CVPR 2022）：[facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
   - Swin（ICCV 2021）：[microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
8. **重建 ML-GCN 邻接**（用训练共现矩阵实时重建，替代 `custom_adj.pkl`）或**换 CLIP 类嵌入**。
   - ML-GCN（Chen et al., CVPR 2019）：[megvii-research/ML-GCN](https://github.com/megvii-research/ML-GCN)
   - CLIP 类嵌入：CLIP（Radford et al., ICML 2021），[openai/CLIP](https://github.com/openai/CLIP)
   > 这两项**已在上一轮实现**（`--recompute_adj`、`--embedding clip`），可直接开启验证。
9. **直接用 mAP 目标**（AP 导向损失）。
   - AP Loss for Visual Detection（Chen et al., NeurIPS 2019）：[cccorn/APLoss](https://github.com/cccorn/APLoss)
   > 因为 mAP 本质是**排序问题**，直接引入可微 AP surrogate 比只调阈值更治本。

### P3 —— 数据与评测
10. **增加稀有类标注（主动学习）**——当前 person/rescue 在 20% 标注子集仅 19–22 个正样本，
    **这是最直接、性价比最高的途径**；优先标注稀有/高不确定性样本（对应 P0 的第 2 点）。
11. **处理域偏移**：可对 6 个相机源做**源归一化/域自适应**，或按源分别训练再融合；避免把相机背景当类别特征。
12. **评价规范化**：上报 **per-class AP**、`util_ratio` 曲线；对**每类**用验证集搜最优 F1 阈值；训练固定 `num_train_iter`、
    统一 `backbone`/`seed`，先跑 `A`（全监督）得到各类 AP **上界**，再看 `D` 与 `A` 的差距集中在哪些类。

---

## 4. 最省事、最可能见效的“推荐实验序列”

1. **跑全监督 A 拿到 per-class AP 上界**（确认稀有类上界 0.5–0.67）。
2. **P0-1 + P0-2**：把 `_sup_loss` 换成有效样本数权重 + `WeightedRandomSampler` 过采样稀有类 —— 预期稀有类明显回升。
3. **P1-4 + P1-5 + P1-6**：放宽阈值（提升 `util_ratio`）+ 置信度软权重 + 加强公平项。
4. **P2-8**：开启 `--recompute_adj` 或 `--embedding clip`（上一轮已实现），验证图头/CLIP 是否带来增益。
5. 若预算允许，**P2-9 换 AP Loss** 直接优化 mAP。

> 每步都建议**只改一个变量**并记录 `per-class AP` 与 `util_ratio`，才能在“多相机、强失衡、小目标”的数据上定位真正有效的杠杆。

---

## 5. 结论（一句话）

数据和代码共同决定了当前缺口：**①20 类里稀有类在 20% 标注下只有 ~20 个正样本 + 小目标缩图丢失细节**（结构性下界），
**②ASL 不适配、类别平衡仅 clamp 5.0 且用无标注预测、阈值过严致 70% 伪标签被丢弃、熵公平项过弱**（代码性瓶颈）。
因此应**优先用有效样本数重平衡 + 稀有类过采样 + 长尾半监督（DASO/CReST）+ 放宽阈值**，
再叠加**更强骨干/多尺度 + 重建 ML-GCN/CLIP**，同时用**每类 AP 与 optim 阈值**来度量每步收益，才能逐步逼近全监督。
