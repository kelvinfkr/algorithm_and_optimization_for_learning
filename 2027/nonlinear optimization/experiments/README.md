# 课程数值实验

这三个实验分别对应第3、4、5讲，共计4学时。Notebook 与命令行入口调用同一套 `run(Config(...))` 实现，避免讲义、交互演示和批量运行使用不同的算法。

| 实验 | 对应讲义 | 交互入口 | 主要问题 |
|---|---|---|---|
| 1 | 随机优化、动量与非凸收敛 | [01_sgd_moments.ipynb](../notebooks/01_sgd_moments.ipynb) | 均值与均方稳定、精确随机回归、batch与学习率 |
| 2 | 预条件、自适应优化与矩阵更新 | [02_optimizer_geometry.ipynb](../notebooks/02_optimizer_geometry.ipynb) | 更新几何、AdamW、矩阵polar、有限步NS与Muon |
| 3 | 参数化、有效几何与跨宽度迁移 | [03_width_transfer.ipynb](../notebooks/03_width_transfer.ipynb) | NTP与mean-field配对尺度、特征变化、验证集迁移 |

Notebook 包含课前预测、课中运行和课后问题，仓库版本保留空输出。默认先运行一个种子的小规模配置；实验2、3另有默认关闭的三种子单元格。单种子和有限候选网格用于机制演示，不能据此建立优化器或参数化的普遍优劣结论。

## 1. 环境与目录

以下命令从课程目录 `2027/nonlinear optimization/` 执行。若当前位于仓库根目录，先进入课程目录；路径含空格，需保留引号。

```bash
cd "2027/nonlinear optimization"
python -m pip install -r requirements.txt
```

安装命令由使用者在终端执行，Notebook 不会自动运行 `pip`。计算依赖由课程的 [requirements.txt](../requirements.txt) 列出，包括 NumPy、Matplotlib 和 PyTorch。Notebook 展示使用已有 Jupyter 或 VS Code Notebook 内核提供的 `IPython.display`，不另依赖 pandas 或交互组件。请选择与安装依赖时相同的 Python 内核。

实验不下载外部数据。实验2、3在本地生成合成回归数据，训练、验证、测试集独立划分；目标的标准化只使用训练集统计量。所有生成文件位于指定结果目录。

### CPU 与 CUDA

- 实验1使用 NumPy，在CPU上运行，没有 `device` 参数。环境报告中的CUDA可用性不等于该模拟实际使用了GPU。
- 实验2、3默认 `device="auto"`：PyTorch检测到可用CUDA时使用CUDA，否则使用CPU，并在 `report.json` 中记录 `actual_device`。
- 使用 `--device cpu` 可以固定CPU；`--device cuda` 要求当前PyTorch环境确有可用CUDA，不可用时明确报错。
- 实验采用便于数值核验的双精度小模型。诊断和评估开销计入部分计时，结果不作为硬件吞吐或生产训练性能基准。

## 2. 用 Notebook 开始

在现有 Notebook 前端打开上表中的文件，按顺序运行。根目录定位代码支持从 `notebooks/`、课程目录或仓库根目录启动；它会定位课程文件并把 `experiments/` 加入 `sys.path`，不使用某台机器的绝对路径。

每个Notebook的第一个配置单元格完整列出本次设置。默认交互规模与脚本默认值不同：

| 实验 | Notebook默认 | 命令行脚本默认 | 保持一致的核心协议 |
|---|---|---|---|
| 1 | 5000条轨迹、20步 | 20000条轨迹、30步 | 单步均方乘子与精确递推 |
| 2 | 宽度16、80步 | 宽度32、200步 | batch=64，每种方法3个候选，默认无权重衰减 |
| 3 | 3个宽度、60步 | 3个宽度、120步 | batch=32，6个共同基础学习率，中心化初始输出 |

交互结果默认写入 `results/notebooks/lab01/`、`lab02/` 或 `lab03/` 的独立种子子目录。修改配置时应同时选择新目录，保留原始对照。输出包含可直接查看的PNG、可导出的PDF、CSV数据和JSON配置报告。

实验2还提供默认关闭的SVD训练对照。打开该开关会在相同配置下使用SVD参考方向，并写入单独目录；它不会把三次NS5误称为精确polar。

## 3. 命令行运行

### 实验1：均值、均方与批量缩放

```bash
python experiments/lab01_sgd.py --paths 20000 --steps 30 --seed 2027 --output results/lab01
```

脚本分为三个部分：

1. 固定 `B=1, eta=0.9`，不加标签噪声，用乘性噪声展示均值收敛与均方发散。
2. 对批量 `1, 4, 16`，比较单步蒙特卡洛二阶矩乘子和精确公式。
3. 取固定样本预算1600及 `eta=0.04*B`，比较含标签噪声的精确二阶矩曲线。`Config.noise_std` 默认0.2，只用于这一部分，不会改变第一部分的无标签噪声设定。

采用输入方差为1的模型时，精确二阶矩乘子是

\[
r_B(\eta)=(1-\eta)^2+2\eta^2/B,
\]

均方收缩要求 `r_B(eta)<1`，等价于 `eta<2B/(B+2)`。不稳定乘性过程的二阶矩可能由少量极端轨迹主导；有限蒙特卡洛漏掉这些轨迹时，经验曲线可能远低于理论期望。这是需要报告的统计现象，不能通过删除极端值来“修复”。

### 实验2：优化器与矩阵更新

```bash
python experiments/lab02_optimizers.py --steps 200 --width 32 --batch 64 --device auto --polar ns --ns-steps 5 --seed 2027 --output results/lab02
```

SVD参考训练可单独运行，保留其他设置：

```bash
python experiments/lab02_optimizers.py --steps 200 --width 32 --batch 64 --device auto --polar svd --seed 2027 --output results/lab02_svd
```

实验采用两层隐藏层的tanh MLP，平均平方损失的一半。每个种子内，所有方法共用数据划分、初始模型、批次索引序列和训练步数。

| 方法 | 候选学习率 | 算法约定 |
|---|---|---|
| SGD+momentum | 0.01、0.03、0.1 | PyTorch SGD，momentum=0.9 |
| AdamW | 0.001、0.003、0.01 | 使用当前实现中PyTorch AdamW的默认矩参数，环境版本会记录 |
| Muon+AdamW | 0.01、0.03、0.1 | 矩阵EMA系数0.95；辅助AdamW学习率为矩阵学习率/10 |

Muon只更新前两层二维权重矩阵。bias和最终readout由AdamW更新；因此这里比较的是已声明参数分工的混合优化器。矩阵方向经过 `0.2*sqrt(max(shape))` 缩放。

课堂NS实现为Frobenius归一化后的三次迭代

\[
X_{k+1}=1.5X_k-0.5X_kX_k^\top X_k.
\]

默认只执行5步，尤其对小奇异值可能与精确polar明显不同。SVD参考对理论零奇异方向采用数值阈值后置零，不任意补成正交基。课堂三次迭代与部分竞速实现使用的其他多项式不同。

默认 `weight_decay=0` 用于隔离更新几何。若显式设置正衰减，SGD中的耦合L2与AdamW/Muon中的解耦衰减并不等价，必须在实验解释中说明。

每种方法按最终验证损失从其三个候选中选择，再评价一次相应测试损失。CSV中 `update_rms` 是实际参数位移的全参数RMS，包含学习率并聚合辅助参数；它不是单个矩阵未乘学习率的更新RMS。计时包含诊断与评估，不用于宣称硬件效率优势。

### 实验3：跨宽度迁移

```bash
python experiments/lab03_width.py --steps 120 --batch 32 --device auto --seed 2027 --output results/lab03
```

默认宽度为 `32, 64, 128`，共同基础学习率为 `0.003, 0.01, 0.03, 0.1, 0.3, 1.0`。通过Notebook的 `Config` 可显式改变宽度或候选；当前命令行入口不提供这两个选项，不能使用不存在的 `--widths` 或 `--rates` 参数。

| 配置 | 前向倍率 | 原始参数SGD学习率 |
|---|---|---|
| NTP | `1/sqrt(width)` | `base_lr` |
| MF | `1/width` | `width*base_lr` |

两种配置都存储原始 `a,w`，并训练平均半平方损失；它们没有把实际readout `b=a/width` 直接存成独立参数。初始函数由冻结的初始参数计算，并在所有输入上从原始输出减去，使初始预测为零，同时保持可训练参数Jacobian不变。

失败配置保留在预声明网格中，JSON使用明确状态和空值，图上标为失败。不要根据已看到的失败结果删除大步长，再把剩余网格描述成原始扫描。`selection_at_grid_boundary` 记录验证选择是否在网格端点；端点最优应触发明确声明的后续扩展扫描，不能被解释为已经找到真正最优学习率。

报告中的 `initial_normalized_probe_kernel_trace` 来自

\[
\Theta_{\mathrm{probe}}=p_mJJ^\top/n_{\mathrm{probe}},
\]

已包含原始学习率倍率和探针样本数归一化。它不是未归一化的训练核，也不是最大特征值。相对漂移分母使用记录在配置中的 `diagnostic_epsilon`，默认 `1e-12`。

特征变化图和过程图使用同一个预声明基础学习率，即候选数组中间索引的值；默认是0.1。它们没有为每个宽度分别挑选最优步长，因而可以检查固定更新约定下的宽度趋势。

## 4. 三个随机种子的后续实验

实验2、3的Notebook均提供：

```python
RUN_MULTIPLE_SEEDS = False
SEEDS = (2027, 2028, 2029)
```

先完成单种子核验，再把开关改为 `True` 并顺序运行后续单元格。所有种子使用同一配置与网格，结果写入 `multi_seed_2027` 等独立目录。汇总只作描述，保留每个种子的验证选择、失败及边界信息。

也可以从课程目录用命令行分别运行：

```bash
for seed in 2027 2028 2029
do
  python experiments/lab02_optimizers.py --seed "$seed" --device auto --output "results/lab02_seed_${seed}"
  python experiments/lab03_width.py --seed "$seed" --device auto --output "results/lab03_seed_${seed}"
done
```

当前脚本的种子同时控制数据采样、参数初始化和批次流。因此多种子统计反映这些因素的共同随机性；若研究目标是固定数据、仅改变初始化，应另行设计并声明实验。每个种子内部仍使用共同初始化与抽样协议比较不同方法。

Notebook中的迁移诊断预先声明绝对验证损失容忍量：小宽度选出的基础学习率，在大宽度上的损失最多比该宽度自身扫描最优值高多少。诊断不使用测试损失调节阈值；“通过检查”也仅指这个有限网格和固定预算。三个种子不能替代更充分的统计研究。

## 5. 输出文件及用途

| 实验 | 文件 | 用途 |
|---|---|---|
| 1 | `lab01_sgd.png/.pdf` | 均值/均方、单步边界、固定样本预算缩放三幅图 |
| 1 | `moment_paths.csv` | 经验矩、中位数与理论矩的逐步记录 |
| 1 | `batch_scaling.csv` | 不同batch的精确有限时间二阶矩 |
| 2 | `lab02_optimizers.png/.pdf` | 验证选择后的曲线、相对更新及下降对齐 |
| 2 | `polar_approximation.png/.pdf` | SVD与不同NS步数的奇异值比较 |
| 2 | `all_curves.csv` | 全部候选的训练/验证、位移、对齐与时间 |
| 3 | `lab03_width.png/.pdf` | 完整验证网格与固定基础学习率的特征变化 |
| 3 | `feature_dynamics.png/.pdf` | 原始参数RMS位移和特征Gram漂移过程 |
| 3 | `width_summary.csv` | 所有候选的状态、初始化与最终诊断 |
| 3 | `training_curves.csv` | 检查点损失、参数位移与Gram漂移 |
| 全部 | `report.json` | 配置、环境、算法约定和关键结果 |

Notebook仅展示其中必要的图和JSON字段，完整数据保存在结果目录。仓库中的Notebook有意不嵌入执行输出，避免将大量图片和重复结果写入源码。

## 6. 每次提交应说明什么

一份可以复核的实验报告应包含：实际运行配置、软件环境与设备、损失归一化、抽样方式、训练预算、候选网格、验证选择规则、随机种子，以及对失败配置的处理。

实验1重点解释理论期望与有限蒙特卡洛的差异。实验2重点区分几何、历史状态和数值近似，说明混合优化器的参数角色。实验3重点检查前向与步长的配对，区分函数学习、隐藏特征变化和统计泛化。

结果与预测不同是一种有效结果。先检查实现约定和假设，再提出可区分的解释；不通过删除失败或极端轨迹、事后更换预算、使用测试集调参来获得预期排序。
