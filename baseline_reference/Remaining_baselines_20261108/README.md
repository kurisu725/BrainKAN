# 三项原文 baseline 补跑（2026-11-08）

本任务先排队，等待 `KAN_crossfit_selection_20261108` 完成训练、评估、汇总校验且进程退出后开始。沿用同一批882人及原五份709训练/89开发/84测试划分，训练种子1。这里是补充实验；五份是随机留出，不是五折。

## 输入

重新核验了已归档的 DPARSF/filt_global/AAL 时间序列。882人的前90个脑区逐人重算 Pearson FC，全部匹配原论文 MAT 矩阵，最大绝对差1.56e-15。此前准备阶段检查的是CPAC目录，现已纠正。时间序列77–315帧，保留全部受试者。

## 模型与固定训练方案

| 模型 | 实现来源和处理 | 训练与选取 |
|---|---|---|
| FBNetGen | 作者代码；GRU、图生成、分组损失、mixup和稀疏正则；embedding=16。有效帧按4帧组成窗口，packed GRU忽略补齐区域；每人余下不足4帧的尾部不进GRU；FC节点特征仍由完整时序得到 | Adam，lr=1e-4，weight decay=1e-4，batch16，500轮最后模型 |
| BrainNetGNN | 作者BrainGNN仓库中的CNN/注意力/门控图网络。适配90ROI；卷积后自适应池化到37个时间位置，160帧时与原编码器相同；修复辅助分类分支未进入优化器的问题；保留原L1正则 | Adam，lr=5e-4，batch32，最多3000轮；按开发目标损失选择模型；学习率平台期10轮、早停15轮 |
| Res-DAE（论文复现版） | 按论文重建ComBat、Fisher前50%特征、EROS-SMOTE、残差去噪自编码器及分类器微调。ComBat使用年龄和性别协变量；全部参数和特征选择仅拟合训练集 | 两组宽度(256,64)/(512,128) × 两个学习率1e-4/1e-3，共4候选。200轮联合训练+50轮冻结编码器的分类器微调；开发AUC选配置，开发Youden指数选分类阈值 |

FBNetGen归一化均值和标准差只来自训练受试者的真实帧。BrainNetGNN逐受试者逐ROI归一化，补齐值不参与前向计算。ComBat的训练参数、年龄/性别效应在开发和测试时固定使用；不使用开发或测试诊断标签进行预处理。

Res-DAE原文没有给出宽度、学习率、训练轮数、dropout和SMOTE邻居数的完整取值。本次明确采用上述4候选、dropout=0.5、邻居数5、补齐到类别平衡；分类阈值用89人开发集选择。原文采用训练内部交叉验证选择阈值。报告会保留这些实现说明，标记为论文复现版。

## 数据职责与输出

共30次候选训练，最终每个模型每份划分只保留一个模型，共15个。30次开发任务全部结束并封存选择后，才统一计算15条最终测试记录。分类结果从保存的逐人预测复算，输出五份逐次值、均值±样本标准差、实现说明和完整原文baseline表。主表保持7个原文baseline和1个BrainKAN（EN共识Top-75）；不自动修改论文。

比较沿用此前已经查看过测试表现的五份划分，报告标注为探索性对照。代码、预设配置和旧文件校验值在启动前冻结，不根据最终测试成绩添加候选或删除划分。

## 安全暂停和恢复

- 双击 `PAUSE.cmd`：等待当前epoch结束后保存一个滚动断点，状态显示 `paused` 后可关机。
- 双击 `STATUS.cmd`：查看等待、训练或暂停状态。
- 双击 `RESUME.cmd`：从断点继续；若前序实验尚未完成，则继续等待。
- 这些按钮只控制本目录的三个baseline。正在运行的交叉选区实验使用它自己目录下的控制按钮。

只保留当前可恢复状态和每模型每划分的一个入选权重。被替换的Res-DAE候选保留训练日志、开发预测、配置和选择记录，不保留所有候选权重。

## 来源

- FBNetGen: https://github.com/Wayfear/FBNETGEN ，commit 3ed6c813920966a40fca9ffca41f90a6384e333c
- BrainNetGNN: https://github.com/UsmanMahmood27/BrainGNN ，commit 70e92da948b0e23bdda84188985653c61a1bc5f6
- Res-DAE: Dai等，*Classification of recurrent major depressive disorder using a residual denoising autoencoder framework: Insights from large-scale multisite fMRI data*，2024，doi:10.1016/j.cmpb.2024.108114。
- neuroCombat: https://github.com/Jfortin1/neuroCombat ，0.2.12。新环境独立安装依赖，原实验环境保持不变。

## 后续核验

`audit/finalize.py` 会核对冻结文件、五份划分、训练预处理来源、开发模型选择、阈值选择、15条测试预测、旧文件校验值和ZIP清单。完成后查看 `comparison.png`，再交付 `COMPARISON.md`、`MAIN_TABLE.md`、`per_partition.csv`、`comparison.json` 和 `BrainKAN_remaining_baselines_20261108.zip`。
