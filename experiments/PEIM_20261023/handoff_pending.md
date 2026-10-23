### Experiment 26：PEIM

状态：`PARTIAL / NO_GO_STAGE_A_INCOMPLETE`。本轮在`/tmp/PEIM_20261023`执行；历史模型重训0；正式训练0/40，GPU0 0/20、GPU1 0/20；未执行fold0正式预检、未产生正式test指标或OOF。

科学问题：在控制影像、年龄性别及site主效应后，检查年龄性别与16维影像表示的乘性交互是否提供训练内ASD/TD证据；比较普通疾病主效应投影与交互投影，再用训练参考疾病原型距离不确定性选择性调制。FiLM和低秩双线性交互本身均为既有工具。审阅MM-GTUNets论文与服务器源码、RTGMFF论文和官方`model.py`、FiLM论文。RTGMFF将年龄/性别FiLM放在视觉特征融合后，直接使用`gamma*h+beta`，使用当前batch年龄均值方差；其论文没有FiLM独立消融。本实验原设计改为32参数零初始化乘性交互、训练条件中心锚定、rho上限0.1、患者级相对L2调制上限5%。

源码核查：原模型为RFE 500维影像、VAE 500维表型及双16维GTUNet；保持871节点转导图、原始分类器和图、eval时VAE采样及GTUNet dropout、RP特殊权重、图构造提前return、注意力权重`torch.tensor`截断梯度与历史RFE行为。调制只计划改动送入multimodal attention的影像特征，D4辅助CE继续读原始影像表示；Graph的模态权重可随attention输入间接变化。

历史机制复盘：M3的非影像疾病原型学生接近随机，teacher/student一致性约0.5，非零梯度不证明疾病语义迁移；PCCI恒定条件残差已显著损害表现，年龄性别残差与错配条件同样偏弱；实验25虽学到约2.53%相对L2的FiLM调制，但F3相对M3的ACC约-0.573pp、SEN约-1.488pp，F3-F4的AUC增益主要由fold2贡献（+5.461pp），去掉fold2后约+0.047pp。实验24训练内site控制年龄性别新增预测较弱并停止于0/40。

预注册四组未变：E1用`P_main`普通疾病主效应投影；E2用控制影像/表型/site主效应后的`P_int`；E3在E2上使用训练参考类原型距离门`g=1/(1+|dist_ASD-dist_TD|/tau)`；E4保持E3，仅在outer split×site内错配年龄性别。统一`q=Z⊙(W_d d)`，`delta=rho*std*(Pq)`，每患者增量限制在原始表示L2的5%。统一损失本为原生CE+Graph和原始历史D4影像原型CE；没有ph CE、KL或其他新损失。该设计未进入正式训练。

阶段A部分完成：历史D4 best checkpoint仅读，fold0权重SHA256=`46a08c2c9ef9b2577b5df1e29a5af9656c3953e1997c9a1a95e64e5f03a1d316`。fold0固定私有RNG `573506739` 下恰执行1次native eval forward；outer-train probe及32份private-fold×site年龄性别联合错配已在内存计算，但方向稳定性汇总遇到`KeyError: 'rank'`，未持久化fold0结果。修复汇总代码后没有重跑fold0；fold1–9各按固定私有RNG恰执行1次forward，结果已保存。TPSSA使用不同诊断seed且没有可替代PEIM probe/null的表示数据。

九个完整折的`Delta_MSE`均≤0，均值`-0.018217`，中位数`-0.017515`；只有1/9折真实值超过同折32份错配Null的P95。交互投影秩在九折的三个private参考内均为2。private参考交互子空间间的平均主角度为58.94度（折均值范围50.57–66.09度，角度较大表示方向差异明显）；交互方向与普通疾病主效应方向的平均夹角为83.51度。以上是9折描述性训练内诊断；fold0缺失使预注册十折证据门不可完整评估。正式训练因此停止于0/40，不能将弱/不完整证据解释为交互假设成立。

冻结前合成验收：30项测试通过；非对称患者纠错方向测试通过（左组纠正2例、引入3例，净-1）；交互梯度中心差分误差约`7.61e-11`；site标签屏蔽后fold0原生图逐元素相同。这里仅是数学/数据边界验收，不构成阶段A通过。

摘要与完整审计：`/tmp/PEIM_20261023/{report.md,summary.json,audit.json,evidence_audit.json,stage_a_failure.json,stage_a_setup.log}`。失败前源码集合SHA256=`15721f9577385916bf7c96e77f65fc05baa054146f2c505a1fb94448fe73770c`；修复汇总并完成fold1–9诊断后的当前源码集合SHA256=`6ee825f9fbe8cf0995a2a8e785759c2eef8c8af6286854829c869dc724f63f32`；当前`audit.py` SHA256=`5cd1637e31fc4ffe4c2885e0e93badb21a4a31d73dd910d645090fe343160935`；部分阶段A结果SHA256=`5bde918104e8a71c2f275b925b72fbd58b7ae2d453e3bdebb963aa18f21289ba`。原始模型源码SHA、28项历史输入/Checkpoint SHA、十折锚点SHA记录在`audit.json`。fold0 `train_start.pt` SHA=`d548b7f377b72ca691c887c0d4abe8a24566d5c2ab308e08e76cca7b8a0f6120`；fold0 split SHA=`354e7670cd6135b6843aef90bbd7f7ec0e3ae7a372df74a8bac198622109eca7`。无正式Checkpoint SHA。正式训练耗时0、worker训练时间0；fold1–9只读诊断运行在PPU0，未记录峰值显存及完整诊断worker统计。执行偏差为fold0审计汇总缺陷造成阶段A不完整，无模型训练或测试泄漏。

科研判断：`NO_GO_STAGE_A_INCOMPLETE`，不能声称PEIM交互证据成立，也不能声称任何分类收益。九折Delta_MSE均非正，1/9超过Null P95，额外交互在本次训练内探针中没有表现为可靠预测增益。后续如继续，需新实验注册，并在执行随机forward前可靠持久化表示及足够恢复信息；不得在本注册中重跑折0或直接启动40折。任何确认仍需独立seed、无泄漏RFE和独立站点验证。

论文参考：[MM-GTUNets](https://arxiv.org/html/2406.14455v3)、[RTGMFF](https://arxiv.org/abs/2509.03214)、[FiLM](https://arxiv.org/abs/1709.07871)。FiLM本身不是新算法；PEIM所候选的任务相关贡献是受主效应控制的疾病条件交互证据与置信选择性调制，本轮未能验证。

handoff append尝试：原文件218804字节，SHA256=`fd2ae885fe9af470fcb1c78f5503d3435e3a8ee9ae8cba98ccc879f9940dccff`；`apply_patch`返回写入失败。失败后复核原文件字节数和SHA不变，结束标记当前计数为0。本正文待追加；本文件SHA记录在`audit.json`。

PEIM_20261023_END
