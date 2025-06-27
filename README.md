该仓库主要存放了《人工智能化学分析的大作业》，仓库框架如下：

NMRnt：原始模型（参考文献：https://doi.org/10.1038/s43588-025-00783-z）

论文详细信息如下：

Title："Towards a Unified Benchmark and Framework for Deep Learning-Based Prediction of Nuclear Magnetic Resonance Chemical Shifts".

Authors: Fanjie Xu, Wentao Guo, Feng Wang, Lin Yao, Hongshuai Wang, Fujie Tang*, Zhifeng Gao*, Linfeng Zhang, Weinan E, Zhong-Qun Tian, Jun Cheng* (* are corresponding authors).

Baseline：在原模型下针对氟化学位移预测任务的微调模型

Baseline_descriptor：加入原子描述符的baseline

UniGATNMR:在下游节点分类头进一步聚合了邻居特征

TempGATNMR：在UniGATNMR的基础上进一步加入了温度调节机制

Traditional_machine_learning:传统机器学习模型

data：F化学位移数据+各模型的数据预处理脚本

如果有任何代码错误，或者对模型有理解不到位之处，请发邮件至yytfoolbetter@gmail.com，您的每一条建议都对初学者的我十分宝贵！！！
