# Experiment 5: PCND_20261016

本目录独立执行六组 PCND 机制检验，不导入历史训练脚本，不改根目录源码、数据或历史产物。P0 只读复用实验1的30折；P1-P6 各3 seeds、10 folds，共180个正式 PPU 训练任务。

正式180/180折及新增450/450次冻结前向已完成，双卡各90折，正式训练失败0。完整结果与限制见report.md、summary.json；当前PCND未获得稳定优于历史基线、容量和错配对照的证据。

## 预注册设计

|组|32维分支输入|可靠性|
|---|---|---|
|P1|train-only PCA 原始影像投影|1|
|P2|训练TD交叉拟合无条件偏差|1|
|P3|训练TD交叉拟合年龄条件偏差|1|
|P4|训练TD交叉拟合年龄/性别条件偏差|1|
|P5|与P4逐值相同的偏差|预注册TD参考密度g|
|P6|Train/Val/Test分区内部同站点联合错配后独立拟合的年龄/性别偏差|错配参考密度g|

每组均为 32-32-1 GELU 分支；末层精确零初始化；每类 logit 校正上限0.25。分支显式作用于同一次原模型 forward 的 logits，不使用 Hook 接入，不改变原融合/Classifier/BN输入。

## 协议与审计

沿用历史871人、ASD=0/HC=1、RFE500、30份划分/初始化/VAE/RNG锚点、Adam1e-4及wd5e-4、最多500epoch及patience100、原CE+Graph、单次随机正式测试。严格保留历史RFE跨折覆盖泄漏、RP循环早返与未注册权重、forward原位投影、VAE重复加载与采样、eval dropout、转导式全队列图。不得解读为无泄漏外站临床泛化。

官方 EarlyStopping 的 save=False，不会写外部checkpoint；相等验证损失重置patience，而最佳checkpoint仍严格按<更新。分支初始化使用CPU私有Generator且fork恢复CPU RNG，不污染PPU RNG。核心hidden=128取根Opt实际默认，不能混用历史PCCI分支hidden=64。

实验4实际仍在 /tmp/PCCI_Attribution_20261016；只读引用并校验，不迁移、不重写。数据产物仍在 /data。若 /data 发生EIO，本轮停止，不自动转移到临时目录。

P6定义在正式训练前由全队列同站点错配，修订为Train/Val/Test分区内部同站点错配。用户已正式授权方案2，不再等待确认。组内私有RNG确定性循环错位，年龄/性别联合移动，不使用标签；非单例无自匹配，单例保持不变。各分区保存排列、SHA256、实际条件改变比例及站点单例。旧30折冲突、旧源码/缓存SHA完整保存在summary.p6_revision及缓存revision。

30份P6重建并验证参考权重不受heldout年龄、性别、影像或标签影响。P1-P5正式features/gate逐值保留；P4/P5辅助错配查询使用同一修订排列，不改变真实TD参考。训练参数不变不意味着heldout推断不变；保留原有转导式人口图。

## 执行

环境 /root/venvs/mm/bin/python；所有模型训练必须在PPU。launch.py先审计、预计算30个CPU特征缓存、执行数值/梯度/边界检查，全部通过后才运行统一180任务队列。每卡最多4worker，单worker1线程，fold%2固定分配，每卡90正式训练任务。安全限流、不杀健康任务、不根据测试表现删选方法。

单fold只有runs/Pn/sNNN_fK.pt；缓存只有features/sNNN_fK.pt。使用原子写入、进程锁、输入和代码指纹校验、成功跳过、最多两次尝试。训练日志只含进度和验证损失，不展示测试指标；180任务全部完成后才汇总正式结果并运行P4/P5/P6固定五次推理。P6只称“错配训练/as-trained”，不假称真实条件。

五次推理与历史PCCI使用相同SHA256前15hex的PCCI-repeat-{base}种子规则。每draw只执行一次core；真实/错配/关闭共享同一native logits，BN严格不变。P4/P5错配只改变query，不重拟合真实TD参考。

P5额外在同一native logits上计算单位gate反事实，描述参考密度权重与患者CE改善的关联；不是额外训练组。实际年龄/性别改变子集单列，不通过跨分区取人提高覆盖率。

主统计先十折平均，再三个seed均值与样本SD。30折配对胜负仅描述，不作为30个独立临床队列。最终通过独立混淆矩阵和rank-AUC复核全部OOF、源文件及产物hash后，生成summary.json/report.md，再追加一次根handoff.md，不自行push。

