# 讲义插图

本目录保存矢量 PDF 和同名 PNG。所有图均由配套 Python 脚本生成，未使用生成式图像工具。

| 文件名（省略扩展名） | 来源 | 内容 |
|---|---|---|
| `quadratic_geometry` | `foundations_demo.py` | 二次模型的两种 GD 步长与 Hessian 预条件 |
| `kkt_geometry` | `foundations_demo.py` | 预算为2的非负二维投影与法向量 |
| `norm_geometry` | `foundations_demo.py` | 同一梯度在不同范数球中的最陡方向 |
| `sgd_stability` | Lab1 记录与 `make_handout_figures.py` | 均值/均方稳定及单步因子的验证 |
| `polar_approximation` | `lab02_optimizers.py` | SVD polar 与有限步三次 Newton–Schulz |
| `width_dynamics` | Lab3 记录与 `make_handout_figures.py` | 固定基础学习率下的参数移动与 Gram 漂移 |

从课程目录运行：

```bash
python experiments/foundations_demo.py
python experiments/lab01_sgd.py
python experiments/lab02_optimizers.py --device auto
python experiments/lab03_width.py --device auto
python experiments/make_handout_figures.py
```

讲义所用训练曲线为 seed=2027 的一次运行。其他种子的结果保存在 `../sample_results/`，不要把单次图形解释为一般性能结论。图片坐标使用英文以减少操作系统字体差异，数学解释在中文图注中。
