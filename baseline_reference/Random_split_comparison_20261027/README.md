# ABIDE 五次不同受试者划分实验

本批实验与原固定划分、NYU站点实验分别保存。冻结计划为 plan.json。

- 数据：保留的 ABIDE 882 人、90 ROI。
- 每份划分：训练709、开发89、测试84人；诊断类别分层，划分种子0–4，开发划分种子1000–1004。
- 这是五次重复随机留出；各次之间允许重复受试者，每次内部三队列无交叉。
- 所有训练随机种子固定为1。神经网络训练50轮并使用最后一轮。
- 共14个配置×5份划分＝70组训练：固定/适配网格的完整KAN、每种网格下三种ROI阈值的BrainKAN、MLP、BrainNetCNN原始logits、FC-GCN、逻辑回归、线性SVM、RBF-SVM。
- 每份划分、每种KAN网格设置均在该份开发集上重新计算10次逐连接置换贡献。三种阈值固定为Direct<0、Min-max<0.5、Z-score<0。
- 每种网格另汇总开发集选择阈值后的BrainKAN结果；选择顺序为开发AUC、较小开发loss、较少ROI、Min-max/Z-score/Direct。最终测试分数不用于选择阈值。
- 经典模型C使用开发集选择；神经网络沿用已声明配置，不按测试结果调整。
- 全部训练/开发和阈值选择完成并冻结之后，才统一评估五份测试集。

运行环境：E:\Brain\BrainKAN\_packaging_audit\venv_cuda\Scripts\python.exe。

后台控制器：run_background.py，按冻结计划运行、汇总、核验并打包。
中断后可重新启动同一控制器；训练按epoch恢复，贡献按90条连接恢复。使用Windows文件锁防止重复执行。

过程文件：status.json、controller_status.json、run.log、run.stderr.log。
结果文件：COMPARISON.md、comparison.json、per_partition.csv、comparison.png、validation.json。
包文件：BrainKAN_random_partitions_20261027.zip（代码、配置、贡献和预测；权重与续训状态保留在runs目录，不放入ZIP）。

原版本文件的校验值保存在previous_artifacts.json；结束时逐一校验。
