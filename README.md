# FreeMatch + ML-GCN + ASL —— 半监督多标签图像分类

本项目将三个开源工作整合为一个**可运行的半监督多标签分类流水线**，用于您的私有数据集
`D:\CV\custom_muti`：

| 组件 | 来源 | 在本项目中的角色 |
|------|------|------------------|
| **FreeMatch** | [microsoft/Semi-supervised-learning](https://github.com/microsoft/Semi-supervised-learning.git) | 半监督训练算法（自适应每类阈值 + EMA teacher + 熵正则） |
| **ML-GCN** | [megvii-research/ML-GCN](https://github.com/megvii-research/ML-GCN.git) | 多标签分类模型（ResNet + 标签相关性图 GCN 头） |
| **ASL** | [Alibaba-MIIL/ASL](https://github.com/Alibaba-MIIL/ASL.git) | 多标签损失函数（不对称聚焦 + 硬裁剪） |

三个开源库没有直接提供"半监督多标签"的组合，因此本项目**提取它们各自的核心逻辑**，
在一个独立、自包含的代码框架（不依赖庞大的 semilearn 运行环境）中完成集成，并在 4 GB 显存的
RTX 3050 Laptop 上做了适配。

---

## 1. 快速开始

```bash
# 1) 安装依赖（建议 Python 3.12 与 CUDA 版 PyTorch）
pip install -r requirements.txt

# 2) 确认数据集位于 dataset/custom_muti（含 data/train、data/val、data/test 及对应标注）

# 3) 训练（默认 ResNet-50、ASL、FreeMatch，labeled_ratio=0.2）
python train.py --c configs/freematch_mlgcn.yaml

# 4) 在验证集评估已保存的模型
python evaluate.py --checkpoint outputs/freematch_mlgcn/best.pth --phase val

# 5) 在独立测试集评估
python evaluate.py --checkpoint outputs/freematch_mlgcn/best.pth --phase test
```

> 若使用更高分辨率（`configs/freematch_mlgcn_resnet50_384.yaml`，384×384）请配合小 batch。
> 更多调参见 [docs/04_训练与验证.md](docs/04_训练与验证.md)。

---

## 2. 目录结构

```
free_mlgcn/
├── README.md                     # 本文件
├── requirements.txt
├── train.py                      # 统一训练脚本（加载/模型/损失/FreeMatch/验证/早停）
├── evaluate.py                   # 统一评估脚本（mAP / F1 等，输出 CSV）
├── run_experiments.py            # 一键跑基准 + 消融矩阵并汇总
├── mlgcn/                        # 核心库
│   ├── models.py                 #   ML-GCN + ResNet GAP/GMP 基线（可配置 backbone）
│   ├── losses.py                 #   ASL / ASL-Soft / BCE
│   ├── freematch.py              #   FreeMatch 多标签核心 + EMA teacher
│   └── metrics.py                #   mAP / macro-micro F1 / hamming
├── data/
│   └── custom_dataset.py         # MultiLabelDataset（含 SSL 标签隐藏）+ collate_fn
├── configs/                      # 训练配置（YAML）
│   ├── freematch_mlgcn_resnet50.yaml
│   ├── freematch_mlgcn_resnet50_384.yaml
│   └── ablation/                 #  基准 / 消融配置
└── docs/                         # 说明文档
    ├── 01_可行性分析.md
    ├── 02_数据准备与数据加载器.md
    ├── 03_模型与算法集成.md
    ├── 04_训练与验证.md
    └── 05_消融实验与基准.md
```

---

## 3. 核心机制（一句话版）

1. **数据**：实体训练/验证/测试集约按 7:2:1 划分；训练集再按 `labeled_ratio` 隐藏部分标签，其余作为`无标签`子集。
2. **模型**：ML-GCN 用 ResNet 提取特征，用标签相关性邻接矩阵（`custom_adj.pkl`）+ GloVe 词向量（`custom_glove_word2vec.pkl`）经 GCN 生成分类权重。
3. **损失**：有标签用 ASL；无标签用"弱增强伪标签 vs 强增强预测"的一致性损失（BCE），并叠加 FreeMatch 熵正则。
4. **阈值**：每类一个自适应阈值 `thresh[c] = time_p · (p_model[c] / max(p_model))`，由无标签置信度 EMA 自适应更新。
5. **教师**：EMA teacher 生成更稳定的伪标签；评估时用 EMA 模型。

---

## 4. 支持的实验与配置

`configs/` 内已提供的 YAML 配置；`run_experiments.py --dry` 可列出全部实验：

| 实验 | 含义 |
|------|------|
| A / B | 全监督 ML-GCN + BCE / ASL（**基线**，用 100% 标签） |
| H | **弱监督基线**：ML-GCN + ASL，只用 `labeled_ratio=0.2` 的标签、**不用无标签**（对比 D 可得半监督增益） |
| C | FreeMatch + ML-GCN，损失换为 **BCE**（验证 ASL 的作用） |
| D | **完整方法**：FreeMatch + ML-GCN + ASL |
| E / F | **无图**（GAP）/ **无 GCN 头**（GMP）+ ASL（验证 ML-GCN 的作用） |
| G_ratio_* | `labeled_ratio` = 0.05 / 0.1 / 0.2 / 0.5（探究无标签数据量收益） |

```bash
# 跑全部实验并汇总到 CSV
python run_experiments.py --base configs/freematch_mlgcn_resnet50.yaml --out_csv results_summary.csv

# 只跑 A、B、D
python run_experiments.py --only A,B,D
```

---

## 5. 依赖与说明

- **PyTorch ≥ 2.0、torchvision ≥ 0.15**（图 / 嵌入加载、RandAugment 均依赖）。
- 图像归一化 mean/std 取私有数据集统计值（`DEFAULT_MEAN/STD`）。
- 代码在 `mlgcn/` 与 `data/` 中各自独立（`models/losses/freematch` 均不依赖 semilearn 框架），便于单独测试与二次开发。
