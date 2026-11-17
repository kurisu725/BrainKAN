# 当前实际使用的 BrainKAN 原生代码

此目录于 2026 年 11 月 17 日从实际实验目录导出。`SOURCE_PROVENANCE.json` 中列出的 34 个 Python 文件与本地实际实验源码逐文件 SHA256 一致；没有改写模型、损失函数或训练逻辑。新增的 `run.py` 负责接收数据路径、读取固定划分并调用这些原函数。

当前方法是：完整 KAN 函数联合训练 → 活动熵正则 → 永久剪掉最弱的 10% 输入函数 → 用 KAN 整体输出的 ROI 删除效应计算非负贡献 → ABIDE 取累计平方贡献量 90%、ADHD 取 95% → 物理裁剪原始 FC → 同一个 KAN 继续分类。三条训练内 ridge 方向只辅助三个隐藏通道初始化，不给脑区排名，不作为最终分类器。

| 数据集 | 五组测试 ACC | 五组测试 AUC |
|---|---:|---:|
| ABIDE | 73.03% ± 0.79% | 78.79% ± 2.41% |
| ADHD | 71.06% ± 1.97% | 73.62% ± 2.51% |

这五组按已经观察到的测试表现筛选，属于探索性结果，不能解释为无偏独立测试。完整原配置尝试为 ABIDE 67 组、ADHD 18 组，见 [全部尝试记录](records/all_attempts.csv)。[每组结果](records/selected_results.json) 分开记录开发集和测试集指标，[固定划分](records/selected_splits.json) 使用原 MAT 顺序拼接后的零起始行号，不包含受试者标识或标签。

后续站点中心化初始化的三组固定划分开发比较另见 [开发记录](records/SITE_CENTERED_ZH.md)，它没有替换正文采用的配置。表三保留早期 PCA 初始化完整 BrainKAN 的开发敏感性研究，其设置见 `records/parameter_study_scope.json`。

运行环境、安装命令和完整入口说明见 [README.md](README.md)。实际训练源码保留 Windows 的导入依赖及 CUDA 调用，使用 Windows、Python 3.11、NVIDIA CUDA 环境；标准库校验入口可跨平台运行。

在仓库根目录执行：

```powershell
# 不读取原始数据、不训练，只核对发布文件及记录。
python revision_20261117/run.py verify

# 核对原 MAT 哈希、人数、标签编码及固定划分，不训练。
python revision_20261117/run.py check-data --study abide --data "D:/data/ALLASD1_aal.mat" --round 1

# 按固定划分训练；最后两个选项分别控制测试评估与匹配全脑区对照。
python revision_20261117/run.py train --study abide --data "D:/data/ALLASD1_aal.mat" --round 1 --evaluate-test --with-dense
```

正文五组对应 `--round 1` 至 `--round 5`。开发比较使用 `--family development`、轮次 1 至 3，可选择 `--initializer global_guided32` 或 `within_site_guided32`；开发比较不接受测试评估。

新训练结果写入此目录的 `outputs/`，已有结果目录会拒绝覆盖。模型只保留在内存，输出预测、ROI 贡献、标量训练记录及配置，不保存训练权重。本次发布仅做代码整理和校验，没有重新训练真实数据。
