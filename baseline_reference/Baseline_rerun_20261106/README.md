# ABIDE 原比较模型补跑

本目录沿用原来的五份受试者随机留出划分：每份709训练、89开发、84测试；相同882人、相同90脑区归档连接矩阵。此轮先运行能直接使用归档矩阵的BrainGNN、BrainGB和BrainNetTF，各5份，共15个模型。每项均保留官方架构与专门损失，并在audit/adapter_changes.json列出接口及AAL90适配。

BrainGNN使用归档FC绝对值作为图边，属于FC输入适配版本；其官方数据管道原用partial correlation建边。BrainGB使用原论文选定的node_concate和concat，hidden_dim=360。BrainNetTF使用两级注意力、两个头、45个池化节点。训练轮数分别100、100、200；所有结果使用最后一轮。各模型自己的学习率、正则化、mixup及调度写入plan.json。没有对测试集逐轮选择最优模型。

全部15个模型开发流程完成并封存后，才统一读取最终测试集。已查看过的五份留出用于本次探索性对照；论文参数实验和最终主方法选择另行确定。

FBNetGen和BrainNetGNN需要时间序列。已有882人的CPAC文件，但目前抽查的重算FC与归档FC不一致，最短78帧。其输入来源及变长处理需要先明确。Res-DAE已找到论文，其原始代码和训练细节仍需定位。没有把其他网络替代成这三项模型。

## 安全暂停

双击PAUSE.cmd，在当前epoch完成和断点落盘后退出。STATUS.cmd显示safe_to_restart=true才可重启；RESUME.cmd明确恢复。训练只保存一个滚动断点，完成后每项模型每份划分保留一个最终权重，共15份，不逐轮保留权重。每轮保留轻量loss/开发指标日志。

旧结果和论文均保持不变。完成时audit/finalize.py验证逐预测指标、开发封存、旧文件校验，并生成COMPARISON.md、comparison.json、per_partition.csv、comparison.png及结果包。
