# CutMix 与 ULDA-inspired 稀有类在线扩增

## 1. 先明确两篇论文解决的问题

### CutMix

CutMix 直接适用于分类训练。对两个样本 `(x_a, y_a)`、`(x_b, y_b)`，从 `x_b` 裁一块矩形贴到
`x_a`，并根据实际矩形面积得到 `lambda`。原论文用混合标签
`lambda*y_a + (1-lambda)*y_b`。

本项目使用 ASL。ASL 的聚焦权重依赖目标值，因此软标签 ASL 不等价于标准 CutMix 目标。代码保留
两份 hard multi-hot 标签并计算：

```text
L_cutmix = lambda * L_ASL(logits, y_a)
         + (1 - lambda) * L_ASL(logits, y_b)
```

类别平衡权重也分别作用于 `y_a`、`y_b` 后再按 `lambda` 合并。

### ULDA

ULDA 原论文不是一个“给少数类生成新文件”的算法。它从无标签辅助集构造 N-way K-shot episode，
把每张原图当作一个伪类，用不同的支持集增强池 `A_S` 和查询集增强池 `A_Q` 制造分布偏移，再用
ProtoNet/MetaOptNet 训练特征提取器。

论文提出的 DSTIM 包含：

```text
TIMadd: lambda = max(u, 1-u), u ~ Beta(0.6, 0.6)
        x_new = lambda*x_i + (1-lambda)*x_j, lambda in [0.5, 1]

TIMsub: lambda = 0.5 + max(u, 1-u), u ~ Beta(0.8, 0.8)
        x_new = lambda*x_i - (1.5-lambda)*x_j, lambda in [1, 1.5]
```

两者都保留锚点标签 `y_i`。原论文支持侧使用 TIMsub、查询侧使用 TIMadd。

当前仓库是 FreeMatch + ML-GCN + ASL 的半监督多标签训练，不是 episodic few-shot。因此代码实现的
是 **ULDA-inspired DSTIM 适配版**：对被采样到的稀有类有标签样本交替使用 TIMadd/TIMsub，保留
现有 ML-GCN 头与 FreeMatch 流程。日志会明确打印这一点，不能把它称为 ULDA-ProtoNet 完整复现。

## 2. 当前数据的长尾情况

训练集共 2209 张、15 类。最少的四类为：

| 类别 | id | 全训练集正样本 | 当前 20% labeled 划分 |
|---|---:|---:|---:|
| person | 9 | 77 | 22 |
| rescue equipment | 11 | 89 | 19 |
| container | 2 | 119 | 30 |
| door | 3 | 134 | 30 |

默认 `rare_aug_quantile=0.25` 会根据**本次 labeled 子集**重新统计并选出计数最少的约 25% 类，避免
把完整标注集的频率错误地套到 20% labeled 实验。为了完全复现，也可以显式设置
`rare_class_ids: [2, 3, 9, 11]`。

## 3. 代码中的实现流程

1. `WeightedRandomSampler` 提高含稀有正标签图像的抽样概率。多标签图像的样本权重取其所有正标签
   类别权重的最大值，避免一个稀有类被共现的头部类稀释。
2. `data/rare_class_augmentation.py` 只选择含稀有正标签的 batch 行作为 anchor。
3. 增强采用“替换 batch 中该行”，而不是追加新行，因此不会把当前 4 GB 显存占用再翻倍。
4. CutMix 让 rare anchor 至少保留 50% 面积/损失权重，并按裁剪后的真实面积重新计算 lambda。
5. CutMix donor 优先共享 anchor 的稀有标签；其余标签由 `y_b` 和精确混合损失处理。
6. DSTIM 优先选择共享稀有正标签的 donor。若 donor 还有 anchor 不具备的正标签，监督损失会把这些
   donor-only 类暂时屏蔽，而不是错误地当成负类。若希望最严格的配对，可把 `ulda_pairing` 改为
   `label_subset`；此时没有安全 donor 的样本会跳过。
7. TIMadd/TIMsub 会先反归一化到 `[0,1]`，混合后裁剪，再按项目 mean/std 归一化。验证集和测试集
   永远不做这些增强。

## 4. 配置项

主配置已经注册以下字段，默认关闭增强以保留历史基线：

```yaml
oversample: False
rare_aug_method: "none"          # none | cutmix | ulda | cutmix_ulda
rare_class_ids: []                # [] = 按 rare_aug_quantile 自动选择
rare_aug_quantile: 0.25
rare_aug_prob: 0.50

rare_aug_pairing: "shared_rare"
cutmix_alpha: 1.0
cutmix_preserve_rare: True

ulda_alpha_add: 0.6
ulda_alpha_sub: 0.8
ulda_operator: "random"
ulda_pairing: "shared_rare"
```

## 5. 推荐运行方式

先单独比较采样器、CutMix、ULDA-inspired 和组合方法。每次使用独立 `save_name`，不要从旧方法的
`last.pth` 续训。

```bash
# 原始基线
python train.py --c configs/freematch_mlgcn.yaml --overwrite "oversample=false rare_aug_method=none save_name=tail_baseline"

# 只开启稀有类过采样
python train.py --c configs/freematch_mlgcn.yaml --overwrite "oversample=true rare_aug_method=none save_name=tail_sampler"

# 稀有类过采样 + CutMix
python train.py --c configs/freematch_mlgcn.yaml --overwrite "oversample=true rare_aug_method=cutmix save_name=tail_cutmix"

# 稀有类过采样 + ULDA-inspired DSTIM
python train.py --c configs/freematch_mlgcn.yaml --overwrite "oversample=true rare_aug_method=ulda save_name=tail_ulda"

# 稀有类过采样 + 两种方法按样本随机二选一
python train.py --c configs/freematch_mlgcn.yaml --overwrite "oversample=true rare_aug_method=cutmix_ulda save_name=tail_hybrid"
```

也可直接运行新增的消融矩阵：

```bash
python run_experiments.py --only D_ssl_mlgcn,I_tail_sampler,J_tail_cutmix,K_tail_ulda,L_tail_hybrid --out_csv rare_aug_results.csv
```

## 6. 如何判断是否有效

训练开始时应看到：

```text
[rare-aug] method=...
[rare-aug] ... selected rare classes ...
```

训练日志中的 `aug / cm / add / sub / ign` 分别表示当前 batch 实际增强总数、CutMix、TIMadd、TIMsub
数量，以及为避免 DSTIM 假负例而屏蔽的 donor-only 标签项数。若使用严格的 `label_subset` 配对后
`eligible` 很多但 `aug` 很少，可提高 `batch_size`、开启 sampler，或改回默认 `shared_rare` 屏蔽策略。

评估报告新增：

- `Tail-mAP`：自动选出的稀有类 AP 均值；
- `Tail-recall`：这些稀有类召回率均值；
- 原有 overall mAP、Macro-F1、Micro-F1 和逐类 AP/precision/recall。

重点比较 Tail-mAP、Tail-recall 和 Macro-F1，同时确认整体 mAP 没有明显下降。建议至少用
`seed=42/43/44` 跑三次后报告均值和标准差，不能只看单次最好结果。

## 7. 重要风险

- ULDA 的关键贡献是 support/query 使用不同增强分布；本项目版本只迁移该思想与 DSTIM，不声称复现
  原论文的无监督 few-shot 结果。
- 多标签随机裁剪、旋转或跨类混合可能制造假负标签。上线训练前应可视化若干 batch，确认领域语义
  没有被破坏。
- 过采样、ASL 类别权重、FreeMatch 尾类伪标签权重和稀有类增强会叠加。若 Tail-recall 上升但 precision
  或 overall mAP 明显下降，应先把 `rare_aug_prob` 从 `0.5` 降到 `0.25`，而不是继续加大权重。
- 当前主配置是 ResNet-101、448、batch 8、uratio 3，比 README 所述的 4 GB 默认设置更重。若显存不足，
  可覆盖为 `backbone=resnet50 img_size=224 batch_size=8`；在线替换增强本身不会扩大 batch。
