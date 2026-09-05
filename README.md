# PR01-01：启动子识别与跨物种泛化

## 1. 项目简介

本项目围绕 **DNA 序列中的启动子识别（Promoter Recognition）** 展开，目标是构建启动子识别模型，并重点研究模型在**未见物种上的跨物种泛化能力**。

启动子识别本质上是根据 DNA 序列判断给定区域是否为启动子的分类任务。传统方法通常在 *E. coli* 等模式生物的数据上进行训练和测试，但不同物种之间在 GC 含量、-10/-35 box 保守性以及物种特异性调控元件等方面存在明显差异。因此，随机划分数据得到的高性能并不一定意味着模型具有良好的跨物种泛化能力。

本项目将从**负样本构造、序列表示方式、模型复杂度以及物种感知数据划分**等角度开展系统实验，分析不同实验设置对启动子识别性能及跨物种泛化能力的影响。

---

## 2. 任务目标

本项目的总体目标是：

> **构建启动子识别模型，在物种感知的数据划分方式下评估模型的跨物种泛化能力，并系统分析负样本构造策略和序列表示方式对模型性能的影响。**

具体目标包括：

* 构建多物种启动子数据集，并设计合理的正负样本构造策略；
* 建立随机划分和物种感知划分的数据评估方案；
* 分析启动子序列中的 **-10/-35 box、GC 含量、序列长度等特征**；
* 使用 k-mer、One-hot 以及 DNA 大模型嵌入等方式表示 DNA 序列；
* 建立 SVM、Random Forest、XGBoost 等传统机器学习基线；
* 进一步探索 CNN、Transformer 和 DNA 大模型等深度学习方法；
* 分析模型在未见物种上的性能下降及其原因；
* 从物种、启动子类型和核心调控 motif 等角度进行错误分析；
* 最终回答不同负样本策略、表示方式和模型结构对跨物种启动子识别的影响。

---

## 3. 研究问题

### RQ1：负样本构造对性能估计的影响

不同负样本构造策略会显著影响启动子识别任务的难度。

本项目比较以下三种负样本：

1. **随机 DNA**：从基因组中随机采样 DNA 序列；
2. **编码区序列**：从蛋白质编码区域中采样；
3. **非启动子调控区**：从已知调控区域中选择非启动子序列。

研究：

> 不同负样本策略下模型性能存在多大差异？哪种策略能够更真实地反映启动子识别的实际难度？

---

### RQ2：序列表示与模型复杂度对跨物种泛化的影响

比较不同 DNA 序列表示方式：

* k-mer；
* One-hot；
* DNA 大模型 Embedding。

同时比较不同模型：

* SVM；
* Random Forest；
* XGBoost；
* CNN；
* Transformer；
* DNABERT-2 等 DNA 预训练模型。

研究：

> 不同序列表示方式和模型复杂度对跨物种泛化能力的影响是否一致？DNA 大模型是否能够学习到更加通用的启动子特征？

---

### RQ3：跨物种场景下的错误模式

模型在不同物种和不同启动子类型上的表现可能存在明显差异。

本项目将进一步分析：

* 不同物种上的 Precision / Recall / F1；
* 不同 σ 因子类型上的识别性能；
* -10/-35 box 保守程度与预测错误之间的关系；
* 不同物种 GC 含量与模型性能之间的关系。

研究：

> 哪些物种或启动子类型最难识别？模型的错误是否与 -10/-35 box 的保守程度以及物种特异性序列特征有关？

---

# 4. 整体技术流程

项目整体流程如下：

```text
┌──────────────────────┐
│      数据获取         │
│ RegulonDB / DBTBS     │
│ 多物种基因组注释      │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│      数据构建         │
│ 正样本：启动子        │
│ 负样本：              │
│ 随机DNA / 编码区      │
│ / 非启动子调控区      │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│      数据分析         │
│ GC含量 / 长度         │
│ -10/-35 box           │
│ 物种间特征差异        │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│      序列表示         │
│ k-mer                 │
│ One-hot               │
│ DNA大模型 Embedding   │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│      Baseline         │
│ SVM / RF / XGBoost    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   深度学习模型        │
│ CNN / Transformer     │
│ DNABERT-2             │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│      核心实验         │
│ 负样本对比            │
│ 跨物种泛化            │
│ 消融实验              │
│ 错误分析              │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│      结果分析         │
│ 性能差异 / 泛化能力   │
│ 物种差异 / motif分析  │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│      回答 RQ1-RQ3     │
│ 完整报告 + 代码 + PPT │
└──────────────────────┘
```

---

# 5. 数据构建

## 5.1 数据来源

项目计划从公开数据库和物种基因组注释中获取启动子数据。

主要数据来源包括：

* **RegulonDB**：用于获取 *E. coli* 启动子及相关调控信息；
* **DBTBS**：用于获取 *Bacillus subtilis* 启动子数据；
* **其他物种基因组及注释数据**：用于构建跨物种测试集。

最终数据集需要至少包含：

```text
sequence
label
species
promoter_type
sigma_factor
source
```

其中：

* `sequence`：DNA 序列；
* `label`：启动子 / 非启动子；
* `species`：物种；
* `promoter_type`：启动子类型；
* `sigma_factor`：σ 因子类型（如果数据源提供）；
* `source`：数据来源。

---

## 5.2 正样本

正样本为经过数据库注释或实验验证的启动子序列。

对于每个启动子，需要根据转录起始位点（TSS）确定统一的序列窗口，例如：

```text
上游区域 + TSS + 下游区域
```

具体窗口长度将在数据探索阶段根据数据分布和相关研究进行确定。

---

## 5.3 负样本

为了研究负样本构造对模型性能估计的影响，本项目构建三种负样本：

### Strategy A：随机 DNA

从基因组中随机采样与正样本长度相同的 DNA 序列。

特点：

* 构造简单；
* 与真实启动子差异可能较大；
* 可能导致模型获得较高但不够真实的性能。

### Strategy B：编码区

从蛋白质编码区域中采样 DNA 序列。

特点：

* 序列来源更加明确；
* 可以避免直接使用随机基因组背景；
* 与启动子存在更加真实的基因组背景差异。

### Strategy C：非启动子调控区

从已知调控区域中选择明确不是启动子的序列。

特点：

* 与启动子具有更加相似的调控背景；
* 分类难度更高；
* 更接近真实启动子识别场景。

---

# 6. 数据划分策略

本项目特别关注**物种感知的数据划分**。

传统随机划分：

```text
所有物种数据
      ↓
Random Split
   ↙       ↘
Train      Test
```

这种方式可能导致训练集和测试集中存在来自同一物种甚至高度相似的序列，因此容易高估模型的泛化能力。

本项目重点采用：

```text
Species A ──┐
Species B ──┼── Training
Species C ──┘

Species D ───── Testing
```

即：

> **训练集和测试集之间不存在相同物种。**

例如：

```text
Train:
E. coli
B. subtilis
...

Test:
Species X
Species Y
```

通过这种方式评估模型面对**完全未见物种**时的识别能力。

同时保留随机划分结果作为对照，以比较：

```text
Random Split Performance
          ↓
Species-aware Performance
          ↓
Generalization Gap
```

---

# 7. 特征与序列表示

项目计划比较三种主要 DNA 序列表示方式。

## 7.1 k-mer

将 DNA 序列划分为长度为 `k` 的连续子序列：

```text
ATGCGTAC

k = 3

ATG
TGC
GCG
CGT
GTA
TAC
```

实验范围：

```text
k = 3, 4, 5, 6
```

统计各 k-mer 的频率或计数，并将其作为传统机器学习模型的输入特征。

---

## 7.2 One-hot Encoding

将 DNA 碱基编码为：

```text
A → [1, 0, 0, 0]
C → [0, 1, 0, 0]
G → [0, 0, 1, 0]
T → [0, 0, 0, 1]
```

因此，一条 DNA 序列可以表示为：

```text
Sequence
   ↓
One-hot matrix
   ↓
CNN / Transformer
```

这种表示能够保留碱基的位置信息，适合用于 motif 识别。

---

## 7.3 DNA 大模型 Embedding

进一步使用 DNA 预训练模型提取序列表示，例如：

* DNABERT-2；
* Nucleotide Transformer。

整体流程：

```text
DNA Sequence
      ↓
DNA Pre-trained Model
      ↓
Embedding
      ↓
Classifier
      ↓
Promoter / Non-promoter
```

重点研究 DNA 大模型学习到的表示是否能够减少物种间的分布差异，从而提高跨物种泛化能力。

---

# 8. 模型设计

## 8.1 Traditional ML Baseline

首先建立传统机器学习基线：

* SVM；
* Random Forest；
* XGBoost。

输入主要包括：

```text
k-mer features
GC content
sequence length
motif-related features
```

这一阶段主要用于建立可解释的 Baseline。

---

## 8.2 CNN

CNN 主要用于捕获 DNA 序列中的局部 motif。

例如：

```text
DNA sequence
     ↓
One-hot
     ↓
Convolution
     ↓
Motif features
     ↓
Pooling
     ↓
Classifier
```

重点关注 CNN 是否能够有效学习类似：

```text
-35 box
   ↓
Spacer
   ↓
-10 box
```

这样的局部序列模式。

---

## 8.3 Transformer

Transformer 用于进一步建模 DNA 序列中的长程依赖关系。

```text
DNA Sequence
      ↓
Embedding
      ↓
Transformer
      ↓
Attention
      ↓
Classification
```

与 CNN 相比，Transformer 更适合分析序列中不同位置之间的依赖关系。

---

## 8.4 DNA 大模型

在资源允许的情况下，进一步使用 DNABERT-2 等 DNA 预训练模型。

实验可以采用：

```text
DNABERT-2
     ↓
Embedding
     ↓
Classifier
```

或进一步进行参数高效微调：

```text
DNABERT-2
     +
LoRA
     ↓
Promoter Classification
```

---

# 9. 核心实验

## Experiment 1：负样本构造策略对比

**对应 RQ1**

固定模型和正样本，仅改变负样本构造方式：

```text
                 ┌─ Random DNA
Promoter +       ├─ Coding Region
                 └─ Non-promoter Regulatory Region
```

比较：

* Accuracy；
* Precision；
* Recall；
* F1；
* ROC-AUC；
* PR-AUC。

重点分析：

> 随着负样本越来越接近真实的非启动子序列，模型性能下降多少？

最终回答哪种负样本策略能够提供更加严格、更加接近真实场景的性能估计。

---

## Experiment 2：跨物种泛化

**对应 RQ2**

采用 Leave-One-Species-Out 或类似物种感知划分。

例如：

```text
Train
├── E. coli
├── B. subtilis
└── Species A

Test
└── Species B
```

分别比较：

```text
Random Split
        vs
Species-aware Split
```

分析：

* 性能下降幅度；
* 不同物种之间的性能差异；
* 哪些物种最难迁移；
* 物种间 GC 含量差异；
* 启动子 motif 差异。

---

## Experiment 3：消融实验

研究不同技术组件对最终性能的贡献。

实验路线：

```text
k-mer
  ↓
k-mer + advanced features
  ↓
DNA Embedding
  ↓
DNA Embedding + Deep Model
  ↓
DNA Embedding + Deep Model + Domain Adaptation
```

比较每个阶段的：

```text
In-domain Performance
        vs
Cross-species Performance
```

重点观察模型性能提升究竟来自：

* 更好的序列表示；
* 更复杂的模型；
* DNA 预训练知识；
* 域适应方法。

---

## Experiment 4：错误分析

**对应 RQ3**

从多个维度分析模型错误：

### 按物种

```text
Species A → F1
Species B → F1
Species C → F1
...
```

### 按 σ 因子

```text
σ factor 1 → F1
σ factor 2 → F1
σ factor 3 → F1
...
```

### 按 motif 保守程度

分析：

```text
-10 box conservation
-35 box conservation
        ↓
Prediction Error
```

重点判断：

> -10/-35 box 越保守，模型是否越容易识别？

同时分析模型错误是否集中在具有特殊启动子结构或物种特异性调控元件的样本上。

---

# 10. 可选实验

根据时间和计算资源，最多选择两个实验进行深入研究。

### Experiment 5：模型架构对比

固定序列表示方式：

```text
Same Representation
        ↓
SVM
CNN
Transformer
```

分析不同模型对于：

* 局部 motif；
* 长程依赖；
* 跨物种特征

的学习能力差异。

---

### Experiment 6：-10/-35 box 影响分析

对已知启动子进行 motif 操作：

```text
Original Sequence
       ↓
Remove / Mutate -10/-35 box
       ↓
Model Prediction
```

比较修改前后的预测概率：

```text
P(promoter | original)
        vs
P(promoter | mutated)
```

用于分析模型是否真正依赖核心启动子 motif。

---

### Experiment 7：物种特异性特征发现

比较不同物种启动子的序列特征：

```text
Species A
   ↕
Species B
   ↕
Species C
```

寻找具有较强物种区分能力的：

* k-mer；
* motif；
* GC 特征；
* 位置特征。

进一步分析这些特征是否与模型跨物种泛化失败有关。

---

# 11. 模型可解释性与可视化

在完成核心实验后，可以进一步加入模型可解释性分析。

目标是回答：

> **模型到底学到了什么？**

例如使用归因方法分析模型重点关注的 DNA 区域，并与已知生物学知识进行比较：

```text
DNA Sequence
────────────────────────────
      ↑              ↑
   -35 box        -10 box
      ↑              ↑
      └──── Model Attention ────┘
```

可以进一步比较：

```text
Model-important regions
          vs
Known promoter motifs
          vs
Known TF binding sites
```

如果模型关注区域与已知 -10/-35 box 或其他调控元件具有较高一致性，可以作为模型具有一定生物学合理性的证据。

---

# 12. 评价指标

主要使用以下指标：

| 指标        | 作用                    |
| --------- | --------------------- |
| Accuracy  | 整体分类准确率               |
| Precision | 预测为启动子的样本中真正启动子的比例    |
| Recall    | 真正启动子中被正确识别的比例        |
| F1        | 综合 Precision 和 Recall |
| ROC-AUC   | 衡量整体分类能力              |
| PR-AUC    | 数据类别不平衡时的重要指标         |

对于跨物种实验，还重点报告：

```text
Performance on seen species
            ↓
Performance on unseen species
            ↓
Generalization Gap
```

其中：

> **Generalization Gap = Seen-species performance − Unseen-species performance**

用于量化模型的跨物种性能下降程度。

---

# 13. 项目实施流程

整个项目按照以下阶段推进：

### M1：问题分析与数据构建

完成：

* 明确分类目标；
* 确定正样本；
* 设计三种负样本；
* 确定物种感知划分方案；
* 完成数据清洗与标准化；
* 确定 RQ1–RQ3。

---

### M2：探索性数据分析

完成：

* 启动子长度分布；
* GC 含量分析；
* -10/-35 box 分布；
* 不同物种之间的特征差异；
* 正负样本分布；
* 数据质量检查。

产出：

```text
EDA Notebook
数据统计结果
数据清洗代码
可视化结果
```

---

### M3：Baseline

完成：

```text
k-mer
  ↓
SVM / RF / XGBoost
  ↓
三种负样本
  ↓
Baseline Results
```

重点得到：

* 三种负样本下的性能；
* Random Split 基线；
* Species-aware Split 初步结果。

---

### M4：完整实验

完成：

* One-hot + CNN；
* Transformer；
* DNA 大模型 Embedding；
* DNABERT-2；
* 跨物种泛化实验；
* 消融实验；
* 错误分析；
* 选做实验。

最终形成完整实验结果表。

---

### M5：最终成果

完成：

* 完整实验报告；
* RQ1–RQ3 回答；
* 可复现代码仓库；
* 实验数据及处理流程；
* 可视化结果；
* 答辩 PPT。

---

# 14. Repository Structure

项目代码建议按照以下结构组织：

```text
PR01-01/
│
├── README.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── scripts/
│   ├── download_data.py
│   ├── build_dataset.py
│   ├── construct_negative_samples.py
│   └── split_by_species.py
│
├── src/
│   ├── features/
│   │   ├── kmer.py
│   │   ├── onehot.py
│   │   └── dna_embedding.py
│   │
│   ├── models/
│   │   ├── svm.py
│   │   ├── random_forest.py
│   │   ├── xgboost.py
│   │   ├── cnn.py
│   │   └── transformer.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── cross_species.py
│   │   └── error_analysis.py
│   │
│   └── utils/
│
├── notebooks/
│   ├── 01_data_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_negative_sampling.ipynb
│   ├── 04_cross_species.ipynb
│   └── 05_error_analysis.ipynb
│
├── experiments/
│   ├── configs/
│   └── results/
│
├── figures/
│
└── report/
    └── presentation/
```

---

# 15. 最终实验逻辑

整个项目最终形成如下研究链路：

```text
              数据构建
                 │
                 ▼
        ┌─────────────────┐
        │ 三种负样本策略   │
        └────────┬────────┘
                 │
                 ▼
          Baseline Model
                 │
                 ▼
        ┌─────────────────┐
        │ 负样本性能比较   │
        │      → RQ1      │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ 多种序列表示方式 │
        │ k-mer / One-hot │
        │ DNA Embedding   │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ 多种模型架构     │
        │ ML / CNN /      │
        │ Transformer /   │
        │ DNABERT-2       │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ 物种感知划分     │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ 跨物种泛化实验   │
        │      → RQ2      │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ 物种 / σ因子 /  │
        │ motif 错误分析  │
        │      → RQ3      │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ 生物学解释与结论 │
        └─────────────────┘
```

---

# 16. 预期成果

项目最终希望得到以下结论：

**第一，明确负样本构造对启动子识别性能估计的影响。**

验证随机 DNA、编码区和非启动子调控区之间是否存在显著性能差异，并确定更加合理的评估方案。

**第二，明确不同序列表示和模型结构的跨物种泛化能力。**

比较传统 k-mer 特征、One-hot 和 DNA 大模型 Embedding，分析模型复杂度提高是否真正带来跨物种泛化能力提升。

**第三，解释模型跨物种泛化失败的原因。**

从物种、σ 因子、GC 含量、-10/-35 box 保守性以及物种特异性 motif 等角度分析模型错误。

最终建立一个：

> **数据构建 → 特征表示 → Baseline → 深度模型 → 跨物种评估 → 错误分析 → 生物学解释**

的完整启动子识别研究流程，并形成具有可复现性的代码和实验结果。
