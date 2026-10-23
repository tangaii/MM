# 实验24：TPSSA

STATUS：**NO_GO_STAGE_A**。正式训练 **0/40**；历史重训 **0**；正式测试 forward **0**。

## 阶段A：共享信息可行性

只读恢复10份真实历史D4 best core及RP特殊权重；每折在其指定PPU上，使用冻结的私有诊断RNG执行一次原生eval forward。保留VAE采样和GTUNet eval dropout；每次仍有871个转导节点，只提取outer-train的16维影像表示。没有使用heldout诊断拟合或评价探针，也没有计算新模型的测试指标。

Site-only和site+age+sex的ridge alpha均为1，使用历史D4完全相同的三个private folds，参考折拟合、query折OOF评价。年龄合法值中位数/标准化及站点类别均限制在outer-train；ridge输入再按各参考集合中心化标准化。历史private-fold分配曾用train标签做平衡，故不声称整个分区过程无标签。

`ΔR²=(SSE_site−SSE_full)/SST`；下表为原始比例，0.002对应新增解释方差0.2个百分点。Null为每外折32次、private-fold×site内年龄/性别联合无标签错配。

| Fold | 真实ΔR² | Null均值 | Null P95 | 真实−Null均值 | min保护范数² | 实际条件改变率 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.002485 | -0.004043 | -0.001346 | +0.001558 | 0.297510 | 98.99% |
| 1 | -0.009303 | -0.003715 | -0.000489 | -0.005588 | 0.189337 | 99.24% |
| 2 | -0.002152 | -0.004055 | -0.000451 | +0.001903 | 0.210321 | 98.98% |
| 3 | +0.000208 | -0.004654 | -0.000068 | +0.004862 | 0.145171 | 99.20% |
| 4 | -0.003374 | -0.003703 | +0.003301 | +0.000329 | 0.220952 | 98.94% |
| 5 | +0.001445 | -0.004842 | +0.000088 | +0.006287 | 0.343228 | 98.85% |
| 6 | -0.003245 | -0.005097 | -0.002185 | +0.001853 | 0.450793 | 99.01% |
| 7 | -0.003250 | -0.003991 | +0.000663 | +0.000741 | 0.215047 | 98.98% |
| 8 | -0.000207 | -0.004704 | -0.000645 | +0.004496 | 0.200146 | 98.21% |
| 9 | -0.000021 | -0.004603 | -0.001385 | +0.004582 | 0.139770 | 98.72% |

## 冻结门槛及停止决定

| 门槛 | 观测 | 通过 |
|---|---:|:---:|
| ΔR²>0至少7折 | 2/10 | 否 |
| 严格超过Null P95至少7折 | 4/10 | 否 |
| 中位数>0.002 | -0.002318430 | 否 |
| 至少7折三个参考集合都有非零有效方向 | 10/10 | 是 |
| 至少7折三个参考方向保护后范数²均≥0.20 | 7/10 | 是 |

三项信息充分性条件失败，因此停止本轮。真实条件相对Null均值较好不能替代真实ΔR²>0及预设规模；非零奇异值同样不能证明有可泛化的共享信息。未执行fold0 train-start训练模式预检，未创建S1–S4模型、优化器或checkpoint。正式启动器已实测拒绝NO-GO后的训练请求。

## 机制解释

真实ΔR²十折中位数为-0.002318，即-0.2318个百分点影像方差的预测误差改善，方向为负。当前固定线性探针没有建立控制site后的足量年龄/性别新增信息。这不证明年龄/性别在所有表示、非线性关系或独立数据中永远无效。
32份Null的实际联合年龄/性别改变率平均为98.91%，最小97.45%；索引改变率平均100.00%，性别改变率25.47%，年龄绝对变化平均5.32岁。重复值没有被计作有效条件变化。该结论针对原始条件Null；S4表示错配未执行。
30个合法参考的共享方向非零，rank值为[2]；方向承载的站点残差影像方差比例均值为24.81%。这只是训练内方向方差，不能写成OOF新增可解释方差。共享/疾病方向绝对余弦范围0.141–0.927，夹角范围21.95–81.89度。保护几何门槛通过并不挽救信息门槛。
三个private参考方向的稳定性在summary.json中按公共原始输出轴的绝对余弦报告。没有正式训练，故参数梯度保护、S2/S3/S4梯度比较、空间塌缩随epoch变化及对齐loss是否恒定均未检验；不把未执行项写成验收通过。

## 论文与服务器代码核查

原论文：[arXiv v3](https://arxiv.org/html/2406.14455v3)；指定仓库：[tangaii/MM](https://github.com/tangaii/MM)。实际服务器代码优先。FC上三角6670维由历史RFE压缩为500维；非影像仅SITE_ID、SEX、AGE_AT_SCAN，经冻结VAE升至500维；两个GTUNet各输出16维，共用由影像与非影像共同形成的融合人口图，之后原注意力融合与16→256→2分类器诊断。ph_unet输出不构成独立于影像的纯人口统计表示。
论文描述分类训练300 epochs、8:1:1和相关距离；服务器历史执行500 epochs、约81:9:10（704/79/88或705/79/87），Torch相似度实现使用欧氏距离。原图edge dropout调用还使用其默认概率。实验24保留服务器协议，没有借机改成论文文字设置。
保留的特殊行为：VAE eval仍采样（mm_gtunets.py:34）；GTUNet F.dropout未传training（gtunet.py:117）；rp_graph.py循环内return（:96，实际行号以源文件为准）；RP权重特殊注册/保存与共享Conv的.data约束（mm_gtunets.py:59、:76）；fusion贡献权重由torch.tensor重建而detach（:133）；原训练跨折覆盖x后的监督RFE污染（train_mm_gtunets.py）；全部871节点转导图；PPU长期轨迹非完全确定。8份指定源码SHA均与开始时一致。

## 实验20–23真实失败机制复盘

- **20_M3**：Non-image student training accuracy ~0.5018, entropy ~0.6926, ph CE ~0.694; KL rises ~0.0012 to ~0.2601. A nonzero KL gradient did not establish learned disease semantics; AUC below D4 and shuffled M4.
- **21_N3**：Age-sex weighted group teacher ~0.4885 ASD probability, cross-patient SD ~0.0169, accuracy ~0.5088. Averaging diluted individual disease information; student stays near chance.
- **22_P1_P3**：Paired matching improves age-sex balance, but embedding-derived coupling does not stabilize improvements; P3-P4 AUC change is fold-sensitive. Patient change direction is reversed in the historical analyzer.
- **23_R3**：Reported age residualization and near-parallel CE gradients are additionally confounded by the verified auxiliary label-order error above. Site fallback is frequent; small AUC differences coexist with worse ACC/SPE.
- **DRPA**：Direct/shared feature alignment and gates did not show consistent independent diagnostic gains; learned projections need not identify useful shared factors.
- **PCCI**：Fusion residuals reduce classification performance; correct correspondence evidence is weak and high perturbation need not help diagnosis.
- **CMCV**：Raw age/sex did not establish stable independent probability-AUC/CE increment; this is not proof of universal absence of useful nonlinear information.
- **LCRA**：Local relation loss is learnable but effective graph/forward probability perturbations are very small; optimization of an auxiliary quantity does not prove an effective prediction mechanism.

**额外核实的实验23实现错误**：其冻结residual.py将query logits按private fold拼接，却使用未重排的原train-order self.y计算残差CE（191–194行）；raw P1对照也在228行重复此问题。10折中实际有46.52%–52.77%的训练诊断目标错配，平均50.19%。冻结manifest和当前历史源码SHA完全匹配，正式run.py调用该类，因此不是只存在于未采用的草稿代码。
独立合成嵌入检查：原实现R3 loss=0.710760，保持同一logits而按query_indices使用正确标签后为0.574774；原raw P1诊断也共用错误顺序。故报告的0.99918不能单独证明它与**正确实现的历史P1**梯度重复。该错误混淆了历史机制解释，但不能据此唯一归因所有正式性能变化。历史源码、报告、模型均未修改，也未重训。
实验22 patient_changes把left候选与right参考的变化反向计数，原合成自测两组完全相同未能发现。新分析器用2人纠正、1人新增错误的非对称样例检验，candidate−reference应为+1；同时验证ASD/TD净变化和fold/label匹配，未改历史文件。

## 预注册S1–S4与正式四指标

完整公式见registration.md，以下仅说明被门槛阻止的研究方案。S1为参考标准化后固定ridge映射的全16维MSE；S2用站点残差age/sex与影像交叉协方差首左奇异向量u（k=1）投影MSE；S3只在新增影像对齐分支采用`h.detach()+(h−h.detach())@(I−ddᵀ)`；S4仅替换private-fold×site内非影像对齐query对应。D4 CE独立保留其5%预算/epoch0..9 warmup/cap0.5；新增image/ph各自5%输出梯度预算，采用独立detach分支尺度。没有旧ph疾病原型CE、疾病概率KL、新MLP、Gate、Margin或InfoNCE。
S1–S4均未训练，无正式四指标、逐折AUC、OOF、checkpoint差异或消融结论。以下只汇总只读历史artifact已有逐折四指标，未对历史logits重新评分，也未参与阶段A规则调整。

| 只读历史组 | ACC % | SEN-ASD % | SPE-TD % | probability AUC % |
|---|---:|---:|---:|---:|
| historical_D0 | 82.1042 ± 6.9547 | 79.4573 ± 10.7591 | 84.4126 ± 5.3796 | 88.1012 ± 6.5145 |
| historical_D4 | 82.5588 ± 6.2440 | 79.9268 ± 9.9731 | 84.8427 ± 5.0503 | 88.6897 ± 6.4295 |
| historical_M1 | 82.5588 ± 5.6516 | 79.6707 ± 8.2702 | 85.0555 ± 5.3004 | 88.5305 ± 6.2626 |
| historical_M3 | 83.1335 ± 6.5898 | 79.9207 ± 9.7059 | 85.9158 ± 6.0239 | 88.5995 ± 6.3386 |

历史表先每折、再10折mean±sample SD(ddof=1)，没有混入seed912/913。fold_metrics.csv与oof.csv仅保留新组输出字段表头，数据行均为0，不能当作完成40折。

## 科研判断与后续边界

科研判断：**NO_GO_STAGE_A**，本轮TPSSA不进入MICCAI 2027第一创新点候选确认。最早失败点是信息充分性：保护可用且存在非零方向，仍没有达到年龄/性别控制站点后的OOF新增解释方差要求。没有证据支持强行投入40折对齐训练，也没有证据称S3优于D4/M3或S4。
后续独立注册应先解决实验23行顺序审计与可重复的信息充分性问题；任何新方法、无泄漏RFE、多seed或独立站点验证都需要另行授权，本轮不新增这些任务。仅对seed911反复设计的结果不支持临床泛化或录用标准判断。

## 完整性、资源、SHA与handoff

阶段A的69项数值/边界检查和14项统计合成检查通过。225份原始源码、输入、历史模型及报告SHA完成终检；冻结manifest和input audit保持不变。正式梯度40项验收与真实train-start预检因NO-GO未执行；本轮不声称S3投影实现已经在真实训练上验收。
GPU0只执行f0/2/4/6/8的5次历史eval；GPU1只执行f1/3/5/7/9的5次历史eval。Stage A双卡墙钟26.01秒；单worker峰值allocated显存约2.605 GiB。小矩阵无梯度ridge/SVD在CPU，未改变原生RNG。所有新文件位于/tmp，这是用户明确指定路径，不冒称写入/data实验目录。
冻结阶段A manifest SHA256：`8b621785aa2cc5c609c88b1fb3ce638cc367c14c47d0ae2991e79ef5287f4679`；config SHA256：`170e023fb16f6bb46ea92694af67822ebc74f7598e3275cdd908a8e77f0797b3`；stage_a.json SHA256：`b0499bacc3ee0c63aa44f80a89bb61cf054c0541845fffb2faff8869cb3a51e6`。各折历史D4 checkpoint、train_start、splits、RFE、VAE、CSV及源码SHA清单见audit.json与manifest.json。新checkpoint数量为0。
handoff追加实际失败：/data返回errno=5 Input/output error。原文件仍为218804字节，SHA256仍为fd2ae885fe9af470fcb1c78f5503d3435e3a8ee9ae8cba98ccc879f9940dccff，旧前缀完全不变；原文件结束标记为0，未伪报成功。待追加正文位于handoff_pending.md，包含唯一结束标记；存储恢复后应先核对原SHA再追加，不需要重训。具体审计见handoff_status.json。
