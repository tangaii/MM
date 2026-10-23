
### Experiment 24 — TPSSA（2026-10-23）

STATUS：NO_GO_STAGE_A。科研判断：NO_GO_STAGE_A。按用户最后指令，全部新增源码与产物在 `/tmp/TPSSA_20261023` 执行和保存；未创建 `/data` 实验24目录。
本段追加前 handoff.md 字节数为218804，SHA256为 fd2ae885fe9af470fcb1c78f5503d3435e3a8ee9ae8cba98ccc879f9940dccff；追加后须核实旧字节前缀不变及结束标记仅一次，实际结果见新实验目录 handoff_status.json。

论文与实际服务器源码核查：已读 arxiv.org/html/2406.14455v3、指定 tangaii/MM 仓库、8份要求的服务器源码及DRPA/PCDCA/PCDPAX/PCDPAX2/PCDPACM/PCRPA/PCCI/CMCV/LCRA代码与报告。实际历史实验23位于 `/tmp/PCRPA_20261022`，指定 `/data` 路径不存在，此偏差如实记录。
FC的6670维上三角经历史RFE压到500维；非影像仅SITE_ID/SEX/AGE_AT_SCAN，VAE输出500维；两个GTUNet各输出16维，共用融合影像/非影像的871节点人口图。因此ph输出不是纯人口统计学独立表示。保留VAE eval采样、GTUNet eval dropout、人口图循环内提前return、RP特殊权重与原位约束、fusion torch.tensor截断梯度、历史监督RFE跨折污染、转导节点和PPU非完全确定性；没有修复原始模型/图/分类器/选模协议。论文300 epochs和8:1:1与服务器历史500 epochs和约81:9:10等差异已记录，服务器历史协议优先。

实验20–23复盘：M3非影像学生准确率约0.5018、entropy约0.6926，非零KL梯度没有证明有效疾病语义；N3条件群体教师概率约0.4885、跨人SD约0.0169，平均化稀释个体信息；P1改善年龄性别平衡，但embedding修正未稳定增加独立收益；R3不能仅凭残差变小认定新增有效信号。DRPA直接对齐无稳定证据，PCCI融合残差降效，CMCV未证实稳定独立AUC增量，LCRA有效图扰动较弱。
额外只读审计确认：实验23冻结residual.py第191–194行按private fold拼接logits后，CE仍用未重排self.y，第228行raw-P1诊断重复同错。十折诊断目标错位46.52%–52.77%，平均50.19%；独立合成同logits检验R3原loss 0.71075952，正确query标签loss 0.57477391。源码SHA与历史冻结manifest一致，正式run.py调用该目标。故原0.99918是两个共享错误标签顺序目标间的余弦，不能证明与正确历史P1梯度等价；不能唯一归因全部性能变化。历史文件未修复、未重训。
实验22 patient_changes方向反转也由非对称样例确认；本轮新分析器以candidate−reference定义，2人纠正、1人新增错误净值为+1，并检查ASD/TD和subject/fold/label匹配。旧分析器未改。

阶段A：seed911，复用历史十折及D4三个private folds。真实best checkpoint由历史anchor与artifact核实，路径 `/data/adodas/MM-GTUNets/experiments/PCDCA_20261019/runs/D4/s911_f{fold}.pt`。完整core及RP特殊状态恢复；每折固定、记录SHA的私有RNG仅一次PPU原生eval forward；仅提取outer-train影像表示。保留871节点转导性，未用heldout标签或测试表现拟合/选规则，未计算任何新正式测试指标。
原始CSV年龄/性别/站点在outer-train编码，site-only与site+age+sex用完全相同private OOF划分和ridge alpha=1；参考统计量中心化/标准化，query从未自拟合。ΔR²=(SSE_site−SSE_full)/SST。每外折32份联合年龄/性别无标签错配，限定private-fold×SITE_ID；不改原始网络输入/图/影像表示，重复值不记作有效变化。

阶段A三项信息诊断：真实ΔR²>0仅2/10（要求≥7）；真实严格高于Null P95仅4/10（要求≥7）；十折中位数−0.0023184295（要求>+0.002）。真实ΔR²均值−0.0022383692，Null均值−0.0043407262，真实−Null均值+0.0021023569，后者不能替代正的OOF新增解释方差。
逐折真实ΔR² / Null P95：f0 −0.0024852623/−0.0013464057；f1 −0.0093025380/−0.0004887150；f2 −0.0021515968/−0.0004507725；f3 +0.0002081107/−0.0000679221；f4 −0.0033744991/+0.0033011623；f5 +0.0014447233/+0.0000881902；f6 −0.0032445346/−0.0021853182；f7 −0.0032500351/+0.0006631689；f8 −0.0002070586/−0.0006452794；f9 −0.0000210020/−0.0013847660。
所有30个合法参考C均有非零有效秩（rank=2，预注册使用k=1），有效方向10/10外折通过；保护后范数²≥0.20按每折三个参考均须通过的冻结规则为7/10，保护几何门槛通过。方向承载训练内残差影像方差均值24.81%，不能写成OOF新增解释方差。共享/疾病方向夹角21.95–81.89度。Null实际年龄/性别改变比例平均98.91%、最小97.45%，索引改变100%，性别改变25.47%，年龄绝对改变平均5.32岁。

三项信息门槛失败，停止本轮，未进入正式40折，正式0/40，train-start真实预检0次，正式测试0次，新checkpoint0份，新OOF0行。不因有方向、存在梯度或相对Null较好强行通过。formal启动器已验证拒绝NO-GO后的训练；完成阶段A结果可resume-skip而不再次forward。

S1–S4预注册完整公式（未实施正式训练）：每个query私折g只用r={pf!=g}的detach表示拟合参考均值/标准差；H_I=(z_I−mean_r)/std_r，H_P=(z_P−mean_r)/std_r；W_g=stopgrad[(H_PrᵀH_Pr+I)^−1 H_PrᵀH_Ir]，alpha=1。S1 L=mean_i,d[H_Ii−H_PiW_g]²。
S2：参考标准化影像和真实age/sex分别以固定ridge去除site，C=R_IᵀR_age,sex/n_r，u=stopgrad首左奇异向量（单位范数，k=1）。c=max(std_r(H_Iu),0.1)；L=mean_i[((H_Ii−H_PiW_g)u)/c]²，不对单维逐人L2归一化。退化方向只能用graph-connected zero并记录，不选随机方向。
S3：同一W/u，d=stopgrad normalize(mean_ASD,r(H_I)−mean_TD,r(H_I))，P=I−ddᵀ；仅新增影像对齐分支用H_safe=H_I.detach()+(H_I−H_I.detach())P；前向loss与S2相同，原生与D4梯度不变，保护仅限标准化影像输出的新增直接梯度。
S4：完整保留S3，固定SHA种子在outer-train/private-fold/site内derangement，只将新增loss的ph query换为另一患者；W/u/d/P、原始ph输入、图、分类器不变。
四组原生CE+Graph与D4均保留。D4原预算5%、warmup epoch0..9、cap0.5；新增影像/非影像各自5%输出梯度预算（image依据保护后真实范数），独立detach分支梯度组合，非普通单lambda自动微分，非参数总梯度5%。无旧ph疾病CE、旧疾病KL、MLP、额外InfoNCE/Gate/Margin。S3显式Jacobian、真实模型保护和S4梯度验收均因阶段A失败未执行，不能声称通过。

历史只读四指标（% mean±sample SD，ddof=1；仅聚合已有保存字段，无额外测试forward）：D0 ACC82.1042±6.9547，SEN79.4573±10.7591，SPE84.4126±5.3796，AUC88.1012±6.5145；D4 82.5588±6.2440 /79.9268±9.9731 /84.8427±5.0503 /88.6897±6.4295；M1 82.5588±5.6516 /79.6707±8.2702 /85.0555±5.3004 /88.5305±6.2626；M3 83.1335±6.5898 /79.9207±9.7059 /85.9158±6.0239 /88.5995±6.3386。S1–S4无正式指标，无S2−S1/S3−S2/S3−S4消融或checkpoint/OOF稳定性结论。

资源与审计：GPU0仅f0/2/4/6/8五次历史eval，GPU1仅f1/3/5/7/9五次历史eval，PPU-ZW810E双卡，阶段A墙钟26.006秒，单worker峰值allocated约2.605GiB；CPU只做小矩阵无梯度统计，不进行CPU神经训练。69项阶段A数学/边界和14项独立统计合成检查通过；225个保护文件SHA终检一致。D0/D4/M1/M3及其他历史组重训均0，RFE拟合0，VAE预训练0，原源码改动0，Git提交/push0。
冻结manifest SHA256：8b621785aa2cc5c609c88b1fb3ce638cc367c14c47d0ae2991e79ef5287f4679；config SHA256：170e023fb16f6bb46ea92694af67822ebc74f7598e3275cdd908a8e77f0797b3；stage_a.json SHA256：b0499bacc3ee0c63aa44f80a89bb61cf054c0541845fffb2faff8869cb3a51e6；输入audit SHA256：4bd7ba8fe5d8e57f12c12ad6e581cfc9cdc5fbe8dac5dce32d9f51a92f113334；输入清单SHA256：c31bf63f5a0554a9051317046b4d8c9c91f512ee2be9b4345b445e7dcbf2df8c。完整源码、逐折D4 checkpoint/core、CSV、train_start、split、RFE/VAE SHA见audit.json、manifest.json、analysis_manifest.json。新checkpoint SHA：无（0份）。
偏差：按用户指令使用/tmp；历史实验23只读定位至/tmp；秩/可保护性采用训练前冻结的三个参考均有效的保守聚合。所有门槛结果后未改规则。报告与详细结果为report.md、summary.json、stage_a.json、history_review.json、audit_final.json，未完成项均明确NOT_RUN。
下一步：不为本轮追加训练/调参。应在未来独立注册中先复核实验23的行顺序与真正新增的信息来源，再考虑固定方法后的无泄漏RFE、未参与设计seed及独立站点确认；本轮不授权这些任务，也不宣称达到MICCAI2027创新或录用标准。

TPSSA_20261023_END
