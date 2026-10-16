# Experiment 4: PCCI Degradation Attribution

用户已授权先在 `/tmp/PCCI_Attribution_20261016` 执行，`/data` 存储恢复后迁至原定 experiments 目录。历史文件只读；不自动 push；最终全部验收后才一次追加根 handoff.md。

## 审计与预注册

2026-10-16 已审计[论文](https://arxiv.org/html/2406.14455v3) 3.2-3.5、4.2、4.4.5/Table 5、七个原始源码及实验1-3指定代码/报告。GitHub `e48c177dc64da4a91dfa44b939b61cb823b65d7b` 的 pcci.py/run.py/config.json/report.md/summary.json 与本地完全一致。作者 HEAD `bb9351b931003535503694ad2f777b6cee1c151f` 的源码唯一差异为既有 CustomConv2d 1x1 兼容替换。

MRRL 使用 RFE 影像、VAE 表型和 AMRS 构造人口图；两支 GTUNet 输出逐患者16维，再由多模态 Attention 融合。年龄、性别、站点已经参与原模型；PCCI 只是现有融合之上的新残差。Table 5 的 ABIDE ACC/AUC：非影像52.00/50.28，影像76.49/85.22，联合82.92/88.21。这不能直接证明患者级条件配对有效。

保留：跨折覆盖 RFE、RP Graph 提前 return、共享卷积 .data 前向约束、非注册 RP 权重、VAE 每次加载和随机采样、GTUNet eval Dropout、edge dropout 默认概率、单次 Linear-ReLU-BN-Linear 分类、CE+Graph Loss、验证选模及 patience100/max500。Tensor cal_feature_sim 实际为欧氏 cdist，注释与论文描述不等同。保持历史行为，不解释成无泄漏临床泛化。

历史 E0/PCCI 共用锚点、划分、VAE与损失；不同之处包括输入预搬运、内存最佳状态缓存、额外残差 autograd 路径及梯度诊断。代码相似或首步等价不足以解释长期结果。本轮独立运行器不导入/调用历史 run.py、launch.py。历史代码只读；PCCI 公式与参数初始化逐项复制验证。

### 源码逐项核对

| 文件/环节 | 实际行为与本轮处理 |
|---|---|
| `model/mm_gtunets.py`：MRRL/VAE | 非影像为3列，VAE潜变量500维；预训练后每次forward重新加载权重并冻结VAE参数，但eval依然重参数随机采样，不用mu替换z。 |
| 同文件：AMRS/人口图 | RP表型亲和图乘以影像与VAE联合特征的相似度；张量分支实际使用欧氏距离。非影像不是仅供新增Adapter使用。 |
| 同文件：GTUNet与融合 | 两条支路逐患者输出16维；Attention融合后仅一次调用原Classifier。图损失仍在两支表示上计算，不把PCCI delta加到原图损失。 |
| 同文件：Classifier | `Linear(16,256)-ReLU-BatchNorm1d(256)-Linear(256,2)`；原训练BN按全体人口图节点统计，eval只读running统计。本轮不改变该转导行为。 |
| 同文件：RP参数 | `.data`约束投影在forward原位发生；多变量共享同一个卷积。RP模态权重经CPU Parameter `.to(device)`后为未注册Tensor，单独导出恢复，未擅自加入optimizer。 |
| `model/gtunet.py` | 原`F.dropout`未传training，因此eval随机性保留；记录TopK节点集合，不改池化或图算子。 |
| `model/rp_graph.py` | 原phenotype循环内提前return、动机边范围保持；使用相同原始函数、历史标签和索引重建，不引入完整三变量修正。历史未单存完整图张量，不伪称有持久化旧图可逐值核对；同RNG历史checkpoint重放用于验证完整前向。 |
| `train_mm_gtunets.py`/`opt.py` | 原RFE覆盖行为、500 epoch上限、patience100、Adam和CE+Graph保留；论文所写300 epoch不覆盖历史正式500协议。 |
| `utils/mydataloader.py` | 使用历史871名ID及缓存标签顺序；ASD=0、HC=1，正式AUC用softmax第0列。原代码以HC为阳性的指标不能无说明混用。 |
| `utils/tools.py` | RFE、随机分层划分、EarlyStopping均核对；无重新特征选择或重新标准化。历史源码输出括号内var与本轮三seed样本SD不等同。 |

### 历史E0与PCCI训练器核对

| 检查项 | E0 / 历史PCCI / 本轮 |
|---|---|
| 核心初始化、VAE、RFE和划分 | 读取同一`train_start.pt`、预训练VAE、selected_features和splits；初始状态、RFE掩码、条件编码及OOF逐项hash/数值核对。图由原函数和已校验输入重建，保存其新指纹，并比较C0/R3同RNG的完整人口图。 |
| 随机状态 | 实例化完成后恢复Python/NumPy/Torch CPU/物理卡本地RNG及cudnn/TF32标志；PCCI独立初始化不消耗core训练RNG。 |
| optimizer | 相同Adam学习率/衰减；core注册参数顺序保持。R0仅core；R1/R2在core后附PCCI；R3仅PCCI。未注册RP Tensor保持原行为。 |
| 优化顺序 | train、zero_grad、一次forward、CE+Graph、backward、step、eval验证、按val CE+Graph选模、原EarlyStopping。无额外对齐或Gate损失。 |
| 诊断与反传 | E0每25epoch retain_grad用于梯度审计，PCCI新增分支改变反传拓扑；R0/R1/R2保留E0 retain_grad。训练中只读取统计，不增加优化步骤。 |
| 输入和保存 | E0每forward传CPU输入并搬运，历史PCCI预搬运至PPU；R0沿用后者，经确定性逐步骤验证等价。E0最佳状态逐次落盘，PCCI与本轮暂存CPU后落盘，不改变val选模条件。 |
| 正式测试 | 最佳状态恢复后保留原始单次随机forward，不采用五次平均替代正式成绩；单独保存正式推理完整RNG。 |
| 长程数值轨迹 | 确定性诊断与原生正式协议分开。首三步相等不代表非确定性图后端500epoch逐位相等；逐epoch存完整RNG指纹、BN计数及损失，完整报告实际R0/R1结果。 |

## 实验

- R0：原模型锚点从头重放，不创建 PCCI。
- R1：相同PCCI与分类器pre-hook，强制零残差；闲置参数进optimizer，无新损失。
- R2：严格C1，仅新增路径输入 imaging.detach()。
- R3：历史C0最佳模型完全冻结，仅训练C3 adapter；每次前向检查全部参数与Buffer，恢复已知RP卷积投影漂移。二阶段验证选模诊断，不与从头训练混为同协议排名。

全部3 seeds x 10 folds x 4组=120折。统一队列，每卡初始4独立进程，每进程单线程；优先同fold物理卡，每卡每组目标15折；自动限流且不终止健康任务。锁与原子保存，校验通过即跳过，每折只保存一个pt。全量新epoch统计存于pt；历史PCCI曲线仅25epoch抽样，禁止插值伪装完整曲线。

训练前30折三步历史E0/R0/R1对照（logits、梯度、Adam、BN、RNG、TopK）、R2 autograd、R3零残差/冻结验证。确定性仅用于等价诊断，正式训练保持原后端。

五次冻结推理复用实验3预注册RNG。D1/D2在同一编码实现和固定eval BN上测尺度、逐层变化和JVP；D4复用旧反事实，只新增R3配对干预；D3读取历史日志并保留分辨率限制。统计每seed先平均十折，再报告三seed均值和样本标准差；30折只是相关配对描述。

### 辅助反事实字段

最终复核发现新运行器原辅助循环在禁用Hook时仍更新`bridge.base`，后续条件因重用该属性而累加了残差。历史PCCI运行器使用局部`base`，没有该问题。120折训练与单次OOF、600次原五draw主预测均正确且保留，不重训、不替换这些结果。

R2/R3每fold追加`counterfactual_audit_v2`，其中`counterfactual_prob_asd`才是最终反事实依据。旧`frozen.counterfactual_prob_asd`和`frozen.base_test_prob_asd`对这两组无效，只为溯源保留。修复用固定局部joint与不触发Hook的原eval分类器；逐draw验证真实条件重现原主预测、R2条件常量不变、R3 off重现C0、BN不变以及原formal/frozen字段签名不变。运行`launch.py`会统一执行/校验这一步，禁止绕过v2直接分析旧辅助字段。

修复过程的早期BN校验曾错取初始化状态为比较锚点，失败产物未写入；现改为加载对应最佳Checkpoint后取锚点。该错误不代表真实BN训练期间漂移。另有seed912/fold6的原生算术概率重放约4.45e-5偏差，经独立原生/确定性前向确认标签和AUC不变，仅诊断按有上界的数值例外处理。详细失败、恢复与原始数值均保存在唯一run.log和summary.json，不隐藏失败记录。

## 执行

`source ~/venvs/mm/bin/activate`

`PYTHONDONTWRITEBYTECODE=1 python -B /tmp/PCCI_Attribution_20261016/launch.py`

仅本目录10个根文件和runs/R0-R3各fold单一pt。summary.json统一汇总，report.md中文报告，run.log唯一全局日志。完成统计、只读校验、报告复核后迁回；全部完成才更新handoff。若存储尚未恢复，明确报告迁移与handoff仍待完成。

## 最终复查

120折训练、600次主冻结推理和全部机制分析已完成。`python -B /tmp/PCCI_Attribution_20261016/analyze.py verify`只读检查文件指纹、正式字段签名、OOF覆盖和独立秩AUC，不构建模型或执行神经网络前向。全部验收后的`launch.py`续跑也只进行该检查，不重新训练或覆盖任何fold。

`/data`仍报Input/output error时，不把迁移或根handoff标为完成。待追加的完整handoff正文暂存`summary.pending_handoff.markdown`；存储恢复后先迁移到指定新目录、逐文件校验，再一次性追加根handoff，保留历史段落。迁移与最终交接前不要清理此临时目录。
