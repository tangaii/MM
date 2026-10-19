# Experiment 12: PC-DCA

独立、预注册的训练期疾病原型监督；不预设有效，不自动提交Git。所有新代码/输出仅在本目录；原始模型与实验1–11只读，最后一次安全追加根handoff。

## 审计

已阅读当前prompt、根handoff、原始8份源码及DRPA、PCND/DualStage、LCRA/Attribution、GBSA/Pilot/Diagnostic实际配置/报告与必要实现，以及PCCI、`/tmp/CMCV_20261019/report.md`。不是根据文件名猜数据。官方commit为bb9351b931003535503694ad2f777b6cee1c151f，8份本地源码仅CustomConv2d最后1×1投影有既有PPU兼容差异，保留.data原位权重约束。

原VAE输入SITE/SEX/AGE，每forward重载和随机采样；双GTUNet输出[871,16]受试者表示。RP循环内提前return、未注册RP权重、共享卷积、tensor重建detach、eval Dropout、原Classifier单次调用及BN、CE+Graph、相等验证损失重置早停且strict best checkpoint均保留。测试恢复best core但不恢复best RNG，保留训练末随机状态与grad-enabled单次测试。

原论文RFE/VAE首先适配500维，不能等于显式语义对齐；Table5包含非影像在图、VAE、融合等处的联合效应，不能直接当作新增正确患者条件增益。[MM-GTUNets](https://arxiv.org/html/2406.14455v3)。[官方](https://github.com/NZWANG/MM-GTUNets)与[当前仓库](https://github.com/tangaii/MM)均已对照。

历史DRPA InfoNCE/可靠性未建立稳定收益；PCCI融合残差退化且正确条件证据弱；PCND发育偏差没有可靠增量；LCRA局部目标学习与实际极小图推理效应不同；CMCV探针未显示稳定非冗余AUC/CE收益但不能证明任何深度互补均不存在；GBSA的FC衍生文本身份匹配未可靠成功。故本轮不加残差/文本/图边修正/新投影网络。

相关主源：[PhenoBrain ICML2026](https://proceedings.mlr.press/v306/meng26f.html)是表型条件长程通信和注意力；[MGDR MICCAI2024](https://papers.miccai.org/miccai-2024/paper/2494_paper.pdf)分离共享/私有图信息；[To Align AAAI2026](https://arxiv.org/html/2511.12121v1)研究对齐和冗余/独有信息结构；[WSSADN MICCAI2024](https://papers.miccai.org/miccai-2024/paper/0336_paper.pdf)研究结构MRI年龄/疾病解耦；[Prototypical Networks](https://papers.nips.cc/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf)已有类别均值距离分类。本轮不能声称首次条件化、原型或疾病监督，仅检验条件疾病差异是否超越全局原型和错误条件。

PhenoBrain的Web PDF解析入口先前失败，后经只读内存下载成功核对正文第4-6页及附录：它使用表型低秩路由修正、多跳ROI传播、文本提示和表型仿射调制注意力，另有脑-表型InfoNCE；非本轮受试者人口图输出上的训练期条件类别原型。其输入还涉及临床描述/量表，不能把其增益直接套用到本项目的年龄/性别/站点。PDF SHA256为`eca2ecb3d980a47be52d42ccd70ce73c22ad18561a37cf98998e4a644680c9a3`。WSSADN正文的球面结构MRI编码、注意力拆分、CGAN年龄指导和一致性目标亦已核对，不等于本轮条件疾病对比监督。此次补充只更新审计说明，不修改已冻结科学定义或训练指纹。

## 固定方法

D0原生新鲜30折；D1全局疾病原型；D2年龄；D3性别；D4年龄×性别；D5对应train同站点错配；D6 D4+站点一致性；D7对应错配。每组30折，总240。D0先完成并审计与历史C0偏差，随后D1–D7交错队列。

train私有3子折，查询的teacher只来自另外两折。教师/坐标统计/类别均值/核/ESS/收缩/可靠性全部detach，只有查询影像z接受梯度；这不是独立OOF深度编码。每类独立核归一，ESS收缩到相同参考集合的全局原型。空类查询不参与辅助CE，无合法query时返回图兼容零。

年龄带宽max(2,IQR(train)/2)；合法年龄(0,120]，缺失仅训练中位数填补，全缺失17岁固定回退；CSV1男2女，未知核.75，同/异1/.5。仅train同站点标签盲联合双射错位，单例不变。训练子折与站点两半使用私有SHA种子，不消费原生RNG。

u按train教师均值/总体SD标准化，SD下限.05；tau=mean_train||u_teacher-mean(u_teacher)||²，下限1，全组一致，未搜索。可靠性两侧是完整互斥站点集合，每侧每类人数≥2、ESS≥2、差异方向范数>1e-8，否则.25。r公式及其他常数见config，训练前冻结。

每个辅助epoch测量native与aux对z的梯度；lambda按预注册5%比例、.5上限及0–9epoch线性热身，detach。完整记录实测加权z梯度比，另每25epoch采样真正影像参数梯度，不能把两者混同。没有新可训练参数。

## 输入和训练

唯一读取DRPA已验收context、3份selected_features/RFE support、30份splits/train_start/fingerprints及对应VAE。读取CSV逐人复核ID/诊断/年龄/SEX/SITE。ASD=0/TD=1，ASD softmax第0列。禁止RFE/VAE重新训练。Adam、原图、native500epoch/patience100及验证规则不改。

环境：`/root/venvs/mm/bin/python`，Torch2.9.0 PPU、NumPy1.26.0、sklearn1.3.2；双PPU，驱动显示每卡98304MiB（96GiB），cgroup32CPU/128GiB。每fold%2绑定物理卡，所有组配对；每卡最多4worker，每worker1CPU线程，保留8GiB GPU，按14GiB/worker预算及cgroup RAM/CPU动态派发。没有CPU神经训练或DDP。

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/venvs/mm/bin/python -B -u experiments/PCDCA_20261019/launch.py --stage preflight
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/venvs/mm/bin/python -B -u experiments/PCDCA_20261019/launch.py --stage baseline
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/venvs/mm/bin/python -B -u experiments/PCDCA_20261019/launch.py --stage methods
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/venvs/mm/bin/python -B -u experiments/PCDCA_20261019/launch.py --stage frozen
```

launch可分阶段`preflight`、`baseline`、`methods`、`frozen`；同配置完成折严格SHA校验后跳过。每折仅一个`runs/Dx/s{seed}_f{fold}.pt`，包含best core/未注册RP/必要曲线/OOF/诊断/冻结结果与内容SHA。没有epoch级半状态续训；未完成从原始锚点开始，不覆盖已验收结果。

5固定锚点连续3步严格比较官方train() AST与D0/aux0的Logits、CE/Graph、梯度、Adam、BN、TopK、完整RNG；确定性仅测试子进程，生产保留原始后端。边界、有限回退和梯度验证集中于run.py，不生成大量测试文件。

formal主结果每seed平均10折，再三个seed均值±样本SD。D0/D1/D4/D5/D6/D7正式结束后固定历史5draw，只作稳定性辅助，不能改选模或替代单次成绩。辅助模块不参与推理，禁止伪造同Checkpoint切换辅助条件的反事实。

最终统计使用必要的独立入口`finalize.py prepare`，验收后才执行`finalize.py append`更新handoff。原因是原分析器直接用float32概率计算CE，而正式指标先转float64；约1e-9的裁剪舍入差触发1e-12验收断言。终检统一为float64，不放宽阈值、不改预测/正式指标/训练源码；finalize自身SHA另行登记。已冻结的config/prototype/run/launch/analyze训练组合指纹保持不变。请使用上述分阶段入口再运行下面的finalize命令；冻结快照中的`launch --stage all`仍调用原分析入口，不是修订后的终检命令。

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/venvs/mm/bin/python -B -u experiments/PCDCA_20261019/finalize.py prepare
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/venvs/mm/bin/python -B -u experiments/PCDCA_20261019/finalize.py append
```

## 可复核性与边界

audit登记受保护源码、数据、历史锚点/Checkpoint/OOF SHA；最终重新读取验证。新产物单文件锁、原子rename、fsync、关闭读回和SHA；只对暂时ENOENT有限重试，EIO失败不当作成功。所有worker日志由单写入者汇入run.log，已完成结果不被活动worker同时读取。

历史RFE跨折监督泄漏、转导871节点人口图、eval随机采样仍存在；本轮只能解释同协议内部差异，不能称无泄漏临床泛化。重复CV的3seed/30fold不是独立外部队列。全部完成或科学性停止后才追加唯一标记handoff并核验旧字节前缀，禁止改写历史结论。
