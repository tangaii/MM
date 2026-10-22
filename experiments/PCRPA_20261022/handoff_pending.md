
<!-- PCRPA_20261022 -->
### Experiment 23: PC-RPA (Phenotype-Conditioned Residual Prototype Alignment)

STATUS: COMPLETE_WITH_DEVIATION. 科学判断：NO-GO for the current MICCAI 2027 first-innovation claim; retain as a mechanism pilot only.

实验文件与产物位于 `/tmp/PCRPA_20261022`。因 OSSFS 在 `/data` 创建新实验目录和追加 handoff 时返回 EIO，本次按用户指示在 `/tmp` 执行。seed=911，fold0-9，R1-R4共40项正式任务；GPU0偶数折20项，GPU1奇数折20项。40/40完成并通过artifact校验，GPU0 20/20，GPU1 20/20；失败0、正式重试0。队列墙钟940.9秒，累计worker时间7060.1秒，峰值每卡4 worker。

方法定义：R1为普通年龄/性别核下混合类别条件均值；R2为同一参考集上ASD/TD条件均值等权；R3为R2与同站点类别平衡均值按min-class ESS beta=(ESS-8)/32裁剪至[0,1]的混合；R4保留真实站点，仅对新增b(v)估计使用D4固定同站点联合年龄/性别置换。R1-R4均保留D4影像CE、M3真实非影像prototype CE、P1真实年龄/性别三元配对权重和相同5%梯度预算；影像辅助为0.5*D4+0.5*残差配对原型CE，不含旧M3 KL。残差D4回退在残差坐标内以D4局部/全局类别权重及ESS/beta构造。

历史只读：D0-D7、DRPA、PCCI、PCND、G1-G4、H1-H4、J1-J4、K1-K4、M1-M4、N1-N4、P1-P4均未重训；特别D0、D4、M3、P1重训数为0。历史D4/M3/P1 OOF和artifact已只读复算。十折train_start、RNG、split、RFE、VAE、模型源码与历史artifact SHA登记在 `audit.json` 和 `manifest.json`。

| Group | ACC % | SEN-ASD % | SPE-TD % | Probability AUC % |
|---|---:|---:|---:|---:|
| historical_D4 | 82.56 +/- 6.24 | 79.93 +/- 9.97 | 84.84 +/- 5.05 | 88.69 +/- 6.43 |
| historical_M3 | 83.13 +/- 6.59 | 79.92 +/- 9.71 | 85.92 +/- 6.02 | 88.60 +/- 6.34 |
| historical_P1 | 83.02 +/- 6.88 | 79.92 +/- 9.56 | 85.70 +/- 6.49 | 88.79 +/- 6.31 |
| R1 | 82.79 +/- 6.15 | 80.68 +/- 8.95 | 84.63 +/- 5.56 | 88.52 +/- 6.47 |
| R2 | 82.44 +/- 6.25 | 79.93 +/- 8.62 | 84.63 +/- 5.56 | 88.51 +/- 6.39 |
| R3 | 82.33 +/- 6.18 | 80.43 +/- 7.94 | 83.99 +/- 6.43 | 88.79 +/- 6.49 |
| R4 | 82.44 +/- 6.34 | 81.68 +/- 9.71 | 83.13 +/- 4.62 | 88.41 +/- 6.34 |

十折mean +/- sample SD (ddof=1)，正式指标仅ACC、SEN-ASD、SPE-TD、probability AUC。

Paired delta (pp，mean +/- sample SD; improved folds): R2-R1为ACC -0.345+/-0.776 (1/10)，SEN -0.750+/-1.208 (0/10)，SPE -0.005+/-1.426 (2/10)，AUC -0.011+/-0.429 (5/10)。R3-R2为-0.114+/-1.478 (3/10)，+0.500+/-1.972 (2/10)，-0.634+/-3.338 (2/10)，+0.276+/-0.673 (4/10)。R3-R4为-0.114+/-1.911 (3/10)，-1.244+/-2.121 (1/10)，+0.865+/-3.070 (2/10)，+0.377+/-0.707 (7/10)。R3-M3为-0.803+/-1.537 (2/10)，+0.512+/-3.678 (4/10)，-1.924+/-1.864 (0/10)，+0.186+/-0.641 (7/10)。R3-P1为-0.690+/-1.454 (1/10)，+0.512+/-3.678 (3/10)，-1.711+/-2.417 (1/10)，+0.000+/-0.704 (5/10)。R3-D4为-0.229+/-1.186 (2/10)，+0.506+/-3.285 (4/10)，-0.851+/-2.497 (2/10)，+0.096+/-0.561 (5/10)。逐折四指标和所有leave-one-fold-out结果见 `fold_metrics.csv`、`summary.json`。

机制与正确性：独立Python嵌套循环oracle核对P1三元组边际；query只使用其他两个private fold作S_g；参考j估计b_j时对角自身权重为零，并且不访问Q_g标签。b、条件核和参考残差停止梯度；query residual保留影像梯度；L_res到z_img有非零梯度、到z_ph无直接梯度。四组P1真实配对权重完全一致。新增可训练参数0；没有KL、InfoNCE或额外分类头。D4 prototype.py与已验收历史文件逐字节相同，SHA256 `ded24a70e72fc1f990d644f88b4445ea0ace750f01629330a9d540853dd44feb`。

fold0同forward预检：D4 CE 0.694661、phCE 0.691890；R1/R2/R3/R4残差CE分别0.693381/0.693353/0.693456/0.693238，历史P1配对CE 0.693478。R1-R2、R2-R3、R3-R4的b平均L2差为0.0203/0.0649/0.2666。R3-R4共同成分确有改变。残差prototype相对P1 prototype平均L2为0.1631，但残差CE与P1 CE差仅-0.000022、梯度余弦0.99918；对D4梯度余弦-0.00798。残差路径不同于D4，但与P1几乎同向，不能宣称独立的P1外训练信号。

R3 site min-class ESS mean10.45、SD8.10、median7.39、p10=3.29、p90=26.38；缺少同站点类别支持0.04%，site beta为0比例54.8%，beta均值0.127。站点贡献偏弱。R2平衡类别没有改善R1。训练内诊断：R3 raw/residual年龄相关性0.0410/0.0129，性别条件L2差0.4872/0.4352，站点between-variance比例0.0434/0.0377，ASD/TD均值间距2.4593/2.4577。提示年龄关联下降且简单疾病均值结构保留，不能识别纯非疾病因子，也不能代表临床泛化。

R4置换年龄/性别平均变更比例98.95%，平均年龄改变量5.28年，性别改变量25.8%；负对照扰动充分。R3相对R4 AUC +0.377pp且7/10折改善，LOO范围+0.212至+0.460pp；但SEN-ASD下降1.244pp。OOF方向经合成记录测试：R4错误转为R3正确16人，R4正确转为R3错误17人；净正确人数-1，ASD净-5，TD净+4。

折级AUC (D4/M3/P1/R1/R2/R3/R4, %)：f0 75.039/74.935/75.195/74.987/74.987/75.039/74.987；f1 91.516/91.251/91.410/91.410/91.357/91.304/90.933；f2 92.895/93.213/93.001/93.160/93.107/93.107/91.251；f3 93.351/92.819/93.351/93.617/93.830/93.457/93.511；f4 85.904/85.957/85.957/85.798/85.957/85.957/86.117；f5 93.883/93.457/93.989/93.723/93.777/93.777/93.617；f6 80.691/80.957/81.436/80.372/80.532/80.372/80.319；f7 88.457/88.457/88.351/88.245/87.819/88.670/88.191；f8 94.734/94.681/94.840/94.574/93.670/94.202/94.574；f9 90.426/90.266/90.319/89.309/90.053/91.968/90.585。f9是R3对R2/M3/P1/D4的最大正AUC折；f6/f8多项历史对比为负。R3-R4 paired checkpoint相对L2均值0.0552，f2/f8/f9分别0.135/0.058/0.345；fold指标、best epoch和validation loss轨迹在summary中。

输入及源码SHA：原始源码 `model/mm_gtunets.py` cb0c27fdc29426e55746a3dc4926ffd2a316d15042fd33a3bd8f207a10e31158；`model/gtunet.py` 30539f14be3b69128610a7bf2a03df4f9ef36a9552002e739ce45558c0d6496c；`model/rp_graph.py` d1c79f3319f9569c8caf98737f3ade96c7c5d334bb97f5c56255ee4ddc3a1783；`opt.py` 068a2fac00ce756eab4a0528a1d5ecc6a4e6e8f4fdd6af87e64fcedb2c18de3a；`train_mm_gtunets.py` 539b4af31189e2215a5a17adcafe78fe2cf4d1d9fafcb5a62aaa8061ce938cba；`utils/mydataloader.py` 7993dc3321c2a6bd2469313371793f1e6210a503a348a6a5d390dc654913ccf5；`utils/tools.py` b08dfd36935b506b90712c5588488956c4a91b670b3bf8ba1186dd14602c3192。context SHA256 `74711ee4a2ae8de9ad275402f2d469906504c2c97ba031887e28cc1a2d2226ab`；raw feature tensor SHA256 `7c8641a0ffbe8e1775d2b8a14c828c3913a3baed2bf28242671c37f553819748`；十折train-start/split/RFE/VAE逐项SHA详见audit。冻结训练代码SHA256 `bf1335fbf7d0b2fd457bbc8733eef5b97bb7918529746b90c00cc87c7251a15c`；manifest SHA256 `274dc79a4b43798cc5f0848581289480e4242ab99497e7669d8dfcc66339bacf`。

训练wall time 940.9秒，累计worker时间7060.1秒；任务失败/正式重试0；每任务一次正式test forward；CPU/PPU softmax最大绝对差1.19e-7，正式PPU概率未被CPU覆盖。原生源码SHA未改变。偏差：总计3次fold0预检forward调用（2通过、1个断言失败），最终冻结版本预检通过后才启动正式任务；该点不是用户要求的单次预检，已如实记录。没有任何正式任务因指标低而重试，没有新增seed或实验组。

建议：不把本轮结果称为MICCAI录用级别创新；不根据本轮test调参。未来如授权，应先锁定方法，另行预注册多seed确认、无监督泄漏RFE及独立站点验证。本实验GO/MIXED/NO-GO判断：NO-GO作为当前主创新，保留为后续机制线索。

handoff追加前长度218804字节，SHA256 `fd2ae885fe9af470fcb1c78f5503d3435e3a8ee9ae8cba98ccc879f9940dccff`，此前结束标记计数0。实际append尝试失败：对 `/data/adodas/MM-GTUNets/handoff.md` 的apply_patch返回 `Failed to write file`。追加后文件仍为218804字节，SHA256及旧前缀SHA仍为 `fd2ae885fe9af470fcb1c78f5503d3435e3a8ee9ae8cba98ccc879f9940dccff`，结束标记计数仍为0；没有部分写入。待追加正文完整保留在此文件中。

PCRPA_20261022_END
