# 凸优化与非线性优化：2027 五讲

**主讲：冯恺睿。**本版将最优性条件、算法收敛和神经网络更新几何放在一条课程主线上，适用于具备微积分、线性代数与基础概率知识的研究生。课程共 **16 学时，其中理论 12 学时、实验 4 学时**。KKT 是完整的第二讲，同时贯穿投影、近端和约束算法。

## 阅读入口

| 讲次 | 讲义 | 理论 / 实验学时 | 主要内容 |
|---|---|---:|---|
| 1 | [凸优化与下降方法](lecture01_convex.pdf) | 4 / 0 | 凸性、光滑性、GD 证明、二次谱、阻尼 Newton、预条件与最小范数解 |
| 2 | [对偶、KKT 与约束优化](lecture02_kkt.pdf) | 3 / 0 | 弱/强对偶、Slater、LICQ、KKT 必要/充分性、活跃集、影子价格、投影/近端、增广 Lagrangian |
| 3 | [随机优化、动量与非凸收敛](lecture03_stochastic.pdf) | 2 / 1 | 条件无偏性、精确随机递推、均值/均方稳定、SGD 驻点界、Heavy Ball、batch 缩放 |
| 4 | [预条件、自适应优化与矩阵更新](lecture04_optimizers.pdf) | 1.5 / 1.5 | 对偶范数、Adam/AdamW、谱范数球、polar、Muon、Newton–Schulz、RMS 与数据方向 |
| 5 | [参数化、有效几何与跨宽度迁移](lecture05_parameterization.pdf) | 1.5 / 1.5 | 有效核、目标方向、矩阵分解、NTK、两层 mean-field/μP 尺度、宽度实验、隐式偏好 |

**[下载83页五讲合订本](course_notes.pdf)**。五份独立 PDF 分别为16、17、16、18、24页（含封面、目录和文献）。每讲均包含学习目标、推导、手算例题、习题与解答要点。五讲共有 **25 个完整例题、32 道习题**；选读部分供课后复习，不要求在课堂逐页讲完。课堂取舍见 [TEACHING_GUIDE.md](TEACHING_GUIDE.md)。

## 文件结构

| 路径 | 用途 |
|---|---|
| `lecture01_convex.pdf` … `lecture05_parameterization.pdf` | 五份独立讲义，分别附参考文献 |
| `course_notes.pdf` | 统一编号、目录和参考文献的合订本 |
| `chapters/lecture01.tex` … `chapters/lecture05.tex` | 五讲正文，可直接修改 |
| `lecture*.tex`、`course_notes.tex` | 单讲与合订本编译入口 |
| `preamble.tex`、`frontmatter.tex`、`references.bib` | 共享排版、课程说明、原始文献 |
| `notebooks/` | 三个交互实验入口 |
| `experiments/` | 可独立运行的实验、公式核验和制图脚本 |
| `figures/` | 讲义使用的矢量图与 PNG 预览 |
| `sample_results/` | 参考运行的配置、CSV、环境和核验记录 |
| `build.py`、`requirements.txt` | PDF 编译与 Python 依赖 |

## 三个实验

1. **SGD 的稳定性与 batch 缩放。**在可精确求解的高斯回归模型中，对照均值与二阶矩的解析递推，观察有限 Monte Carlo 对稀有轨迹的遗漏。实验使用 NumPy，在 CPU 上即可完成。
2. **SGD、AdamW 与教学版 Muon。**固定模型初始化、数据划分和 batch 序列，每种方法使用三个预先声明的学习率配置。记录验证损失、更新 RMS、相对更新和下降方向夹角；测试集只在配置选定后评价。教学版 Muon 使用隐藏矩阵上的 EMA + 三次 Newton–Schulz 近似，偏置与读出层使用 AdamW。具体约定见脚本与第四讲。
3. **宽度、参数化与特征学习。**比较 NTP 的原始 SGD 步长 `eta` 与两层 mean-field 的 `m*eta`，减去冻结初始函数以控制初始预测。记录原始参数移动、特征 Gram 和归一化探针核的变化。完整保留扫描网格与失败配置，标记最优候选是否位于网格边界。

详细实验协议及 Notebook 链接见 **[experiments/README.md](experiments/README.md)**。全部数据由代码本地生成，无需下载数据集或申请账户。

### 环境与运行

在本目录打开终端。路径含空格，进入目录时请加引号：

```bash
cd "2027/nonlinear optimization"
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python experiments/foundations_demo.py
python experiments/lab01_sgd.py
python experiments/lab02_optimizers.py --device auto
python experiments/lab03_width.py --device auto
python experiments/check_math.py
python experiments/make_handout_figures.py
```

也可运行 `jupyter lab` 打开 `notebooks/`。PyTorch 实验的 `auto` 优先选择可用 CUDA；`--device cpu` 可强制使用 CPU，`--device cuda` 会在 CUDA 不可用时给出明确错误。GPU 环境需安装与本机驱动匹配的 PyTorch 版本。参考运行在 CPU 上完成，具体环境写在每份 JSON 中。

正式实验报告至少使用三个种子，例如分别运行 `--seed 2027`、`--seed 2028`、`--seed 2029`，并为每次运行设置不同的 `--output` 目录。Notebook 提供默认关闭的多种子执行开关。一次默认运行是教学示例，不构成优化器排名或渐近理论的证据。

`results/` 为本地生成目录，不纳入版本控制。已发布的 [sample_results](sample_results/) 是参考记录，包含种子、配置、运行环境与全部候选状态；计时包含诊断和评价，不作为硬件性能基准。

### 编译 PDF

需要 **XeLaTeX、latexmk、BibTeX、xeCJK 与 Fandol 字体**。完整 TeX Live 安装通常包含这些组件；精简安装需补中文语言包和常用 LaTeX 扩展包。

```bash
python build.py
# 只编译第二讲：
python build.py lecture02_kkt
# 只编译合订本：
python build.py course_notes
```

若只修改正文，不需要重新运行实验。只有更新讲义插图时才需要生成相应结果，再运行 `experiments/make_handout_figures.py`。`--tex-support PATH` 可用于指定单独解包的 xeCJK/Fandol 目录，正常安装无须使用。

## 本版的分析约定

- **条件先于结论。**凸 KKT 的充分性、Slater 下的乘子存在、非凸 LICQ 下的必要性分别陈述。
- **标明结论类型。**区分精确恒等式、梯度流或小步长近似、有限实验观察和研究问题。
- **损失与坐标显式化。**区分总损失与平均损失，原始参数与实际前向系数，SGD 与 Adam 的步长换算。
- **稳定、收敛与泛化分别讨论。**小梯度、均方有界、找到插值解与测试误差好不是同一个结论。
- **实验保留完整协议。**使用独立训练/验证/测试划分，记录扫描范围、失败配置、seed 和诊断量。

正文引文直接指向教材、论文或作者技术说明，详见 [references.bib](references.bib)。材料完成了交叉数学审阅、六个数值核验模块、Python 实验运行和 PDF 编译检查；验证记录见 [sample_results/VALIDATION.md](sample_results/VALIDATION.md)。

## 旧版材料

原仓库的全部材料保存在 **[2026 目录](../../2026/)**。归档保持原文件内容和相对目录关系，新讲义的修订均发生在本目录。
