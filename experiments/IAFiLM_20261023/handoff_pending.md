
<!-- IAFiLM_20261023 -->
## Experiment 25：IA-FiLM（Identity-Anchored Phenotype FiLM）

状态：COMPLETE；40/40正式任务，GPU0=20/20、GPU1=20/20；历史组重训0；失败0、重试0。唯一seed=911，ABIDE历史十折，ASD=0（403人），TD=1（468人）。实际执行目录`/tmp/IAFiLM_20261023`；创建计划目录`/data/adodas/MM-GTUNets/experiments/IAFiLM_20261023`时遇EIO errno=5，产物待存储恢复后按SHA迁移。

RTGMFF官方`model.py`在HWM/CSTE视觉融合后、ASAM前执行FiLM，直接生成无界gamma/beta并用当前batch年龄均值/标准差；本实验借鉴条件调制问题，采用outer-train年龄标准化、年龄性别输入、训练参考锚定和有界残差式恒等FiLM，不将FiLM本身宣称为新算法。PCCI中C0约81.88% ACC/88.54% AUC，C1常量残差77.78%/85.25%，C3为77.63%/85.07%，C6错配为77.44%/84.98%；常量残差的损害提示条件无关扰动也会破坏表示。历史M3约83.13% ACC、79.92% SEN、85.92% SPE、88.60% AUC；ph学生疾病准确率约50%，KL虽有梯度但未建立有效疾病语义。本轮复用PCDPAX `alignment.py` 与PCDCA `prototype.py` 的已验收M3/D4目标，没有使用实验23的residual实现。

统一FiLM：`d=[age_z,male,female]`，年龄用outer-train合法值中位数填补、train均值/总体标准差标准化并clip[-3,3]，未知性别为[0,0]；MLP为Linear(3,16)-GELU-Linear(16,32,bias=False)，末层全零，共576参数。`delta=f(d)-f(d_ref)`，`gamma=1+rho*tanh(delta_gamma)`，`beta=rho*tanh(delta_beta)`，`h'=gamma*h+beta`，`rho=0.1*min(epoch/9,1)`。因此训练条件均值严格恒等，gamma在[0.9,1.1]，beta在[-0.1,0.1]。

F1：融合后joint_embed、classifier前FiLM，M3 D4+真实ph CE+0.25 KL。
F2：融合前只调制img_embed后送mm_attention，D4和Graph仍捕获原始img/ph输出；保留M3 KL。
F3：同F2，唯一目标差异是移除KL，保留D4与真实ph CE，为主要候选。
F4：同F3，仅FiLM条件在train/validation/test各自的outer split×SITE_ID内无标签derangement；原始年龄性别、VAE、图、D4、ph CE及分类目标不变。

MM-GTUNets服务器源码审计确认：RFE 500维，人口学输入为SITE_ID/SEX/AGE_AT_SCAN，VAE 500维，双GTUNet各16维；Graph使用未调制嵌入，但注意力权重会随F2前调制改变，因此Graph实际前向值可能间接受影响。保留了转导871节点图、历史RFE行为、VAE随机采样、eval dropout、RP特殊状态及融合注意力历史梯度截断。D4和ph CE读取原始嵌入。训练、选模、测试协议按预注册执行。

预训练验收：31项FiLM数学/Hook/梯度测试、24项历史M3数学测试、非对称患者纠错测试通过；seed911/fold0 train_start单次真实forward通过，未计算test指标。FiLM参数梯度非零；注意力/Classifier路径有效；零初始化严格恒等；M3 KL仅F1/F2计算。F4无标签置换的条件索引变化比例train/val/test为100.00%/96.58%/97.48%，联合条件改变98.85%/95.57%/96.33%，年龄改变98.64%/95.32%/95.75%，性别改变约25.22%/25.06%/25.02%。F4真/错条件在同一已捕获test表示上的相对特征L2差约2.340%。

四指标为十折均值±sample SD（ddof=1）：
|组|ACC|SEN-ASD|SPE-TD|probability AUC|
|---|---:|---:|---:|---:|
|D0|82.10±6.95|79.46±10.76|84.41±5.38|88.10±6.51|
|D4|82.56±6.24|79.93±9.97|84.84±5.05|88.69±6.43|
|M1|82.56±5.65|79.67±8.27|85.06±5.30|88.53±6.26|
|M3|83.13±6.59|79.92±9.71|85.92±6.02|88.60±6.34|
|F1|82.22±6.35|78.70±10.14|85.27±4.64|88.64±6.03|
|F2|82.10±6.27|78.43±9.22|85.28±5.33|88.77±6.18|
|F3|82.56±6.07|78.43±8.75|86.13±5.51|88.72±6.18|
|F4|82.22±6.32|79.68±9.63|84.41±5.10|88.13±6.01|

配对均值差（百分点，前组减后组；括号为四指标ACC/SEN/SPE/AUC）：F2-F1=(-0.118,-0.262,+0.009,+0.139)，AUC正向6/10折且LOO范围+0.071至+0.178；F3-F2=(+0.460,-0.000,+0.851,-0.058)，AUC正向1/10折；F3-F4=(+0.345,-1.244,+1.721,+0.588)，AUC正向6/10折、LOO +0.047至+0.725；F3-D4=(+0.001,-1.494,+1.286,+0.026)，AUC正向6/10折、LOO -0.101至+0.153；F3-M1=(+0.001,-1.238,+1.073,+0.186)，AUC正向7/10折、LOO +0.106至+0.253；F3-M3=(-0.573,-1.488,+0.213,+0.117)，AUC正向5/10折、LOO +0.029至+0.248。F3相对D4/M1/M3均未实现完整四指标改善，取消KL也未改善AUC。

机制：F3 heldout调制相对表示L2均值约2.531%，gamma约1.00020、beta约-0.000049；F1/F2/F3/F4各576参数，训练参考条件均严格恒等。FiLM可学习到非零条件调制，但F3相对F4敏感度下降1.244pp，净正确仅+3人（ASD净-5、TD净+8），故不支持稳定ASD识别增益。F3相对M3净正确-5人（ASD净-6、TD净+1）；F3相对D4净正确0（ASD净-6、TD净+6）。F2-F1同折best epoch相同；F3-F2仅1折best epoch不同；F3-F4仅1折不同。位置比较的AUC有小幅一致正差，但Graph/注意力权重不是强制恒定，必须作为间接机制报告。

OOF：每组871人恰好一次，四组3484行；历史D0/D4/M1/M3只读OOF亦完成ID、fold、标签、test顺序核对。患者纠错方向通过非对称合成测试。分析从保存的单次PPU test logits/概率复算四指标，没有额外模型forward或依据test选择模型。D0/D4历史指标字段由只读包装器显式映射为正式四列；分析器问题不导致重训。

正式训练墙钟665.70秒，launcher累计worker经过时间4800.44秒，40任务训练时长合计4011.36秒；GPU0/GPU1各20任务，峰值分配显存约8096MiB，重试0、失败0。历史D0/D4/M1/M3及其他历史模型重训0，RFE重拟合0，VAE重预训练0。实验源码/config/manifest SHA已冻结；manifest SHA256 `0ebf68b28a106162a4c2ddf5d18b02a9f6d16bd984de392498abc8e1ab1dc5d5`，training code aggregate SHA256 `5941bb34ffb953f0f1d04a90c246a3103c9f4d7ed6cf784e80c542d2a4c6989c`。训练文件SHA256：config `0cf32aef71d1a37da0fd8cd3770cb938515246f5502b9f37d78e5dfbc2e6f29f`；film `c2aecef5f4d3944078a34bf744bef943ab049a4c369c25bbc567526c9fc68692`；test_iafilm `0a841a6f354fd41b4a7738773a5f8334cbdd46af1a6257a87775bd461a579bcc`；test_analysis `669c5fcca1c14aa749a4e6d20db84013c7ba1cd8653c4232e5eafd4c14e45a92`；run `20ce175961a4e4d314dbd643c7d090956a680bfd90336aa611ebb178fcc9fb16`；launch `8eeab47368b0cef6ec3a9d22e1da4e7ec550f3a71e55a4ff78f9795c0ad4455c`；analyze `31227074d6917cb8b14fb7a39042ef1355e4acc088e456ecc8367bacc4e8483f`。

原始项目源码SHA256：opt `068a2fac00ce756eab4a0528a1d5ecc6a4e6e8f4fdd6af87e64fcedb2c18de3a`；train_mm_gtunets `539b4af31189e2215a5a17adcafe78fe2cf4d1d9fafcb5a62aaa8061ce938cba`；mm_gtunets `cb0c27fdc29426e55746a3dc4926ffd2a316d15042fd33a3bd8f207a10e31158`；gtunet `30539f14be3b69128610a7bf2a03df4f9ef36a9552002e739ce45558c0d6496c`；rp_graph `d1c79f3319f9569c8caf98737f3ade96c7c5d334bb97f5c56255ee4ddc3a1783`；mydataloader `7993dc3321c2a6bd2469313371793f1e6210a503a348a6a5d390dc654913ccf5`；tools `b08dfd36935b506b90712c5588488956c4a91b670b3bf8ba1186dd14602c3192`；metrics `59405483453a068143de753301910250f2609fa148fd4a42d4cf56ebcc432056`。历史M3 alignment SHA `c4d81afd44fc765f6274727800aea90a4a2bf5777ac74c100adfb68fba4f1`；历史PCDCA prototype SHA `ded24a70e72fc1f990d644f88b4445ea0ace750f01629330a9d540853dd44feb`。

输入SHA256：context `74711ee4a2ae8de9ad275402f2d469906504c2c97ba031887e28cc1a2d2226ab`；phenotypic CSV `009f01a89697113e5375ded4fbcff4c85e89a4daa6d13e63c21ae5d57dda0865`；id.txt `7cb85ff412ed55532c04d6576dba04a8a7dd73206f351c0bf0aae09dc01a16a5`；up_triangles `7c8641a0ffbe8e1775d2b8a14c828c3913a3baed2bf28242671c37f553819748`；labels `a0d8f1d34f66444f6dc5df78ab11035965e3d0d0de198f725d01e01b2a396886`；ph_data `27c9cd94988bc9b11b0d583aca70bf2e1e175d391ac83945bb0c15267c0bcca4`；ph_dict `e1e30ecf4a2401e18270f334532bc8211be14571fb78b6f6e3eb66c6cb91fade`。十折train_start、split、fingerprint、selected feature、RFE support及VAE checkpoint逐项SHA见`/tmp/IAFiLM_20261023/audit.json`（SHA256 `86272e3be48f36c3538f14d84a07cb9d9575e90a76ef471b790b2e3d4c365af3`）。

40份正式Checkpoint SHA256，fold顺序f0..f9：
F1 `ef6b3de641e4e72fbac736db75dd1de11ee2ee844daa55c8217d47a670189c08 8ea393d30acef098193850da2e8fe26607bade7b20af5972aa892eee60b6288a eb8324f1d7ae2b04526841133cbb85bef580a720d25011b81abd4f569fa296ca 5b8d4b78b283e3e8730beb539905663d1d3632f9556bed625453045f2fb37513 4535da209f3ebc190911c56f378ad875622857de2d84e85c570067b1c3f84dc6 0770ee372de379c43e4914d3f482d199b2fd466e5409c6bf6965ee742789bb1f b21947633981cbc00a91f8a0b809658f5e770a34c841d0dcb51fd2eb91ae7e20 9bfded399fef5eb2bfa00ddad12e764b9e113830e2c4a37fd6edebca991ad3ea d5c718b544633d9b6fa1598acc106f19c3bf987325cfa88c4f5cfaa2abe990a8 b9d6df7d7dd6a70ec6043721126b623c768151c40dbc4e2fadab2267348a26da`
F2 `6988f336b354c6d1673881b5f28469b17164da2d5428c62873701e8de6835a08 779a26f2394b37967e78d29e1e85cc4ed5757043223d6d14a399ac2afbd1ad22 66398ea5c6fbce52f3bdc7dcaa95dd575bc41ac0e2f96ab18b41d4b29d9a2de0 719803f77e7bbee6667c8091514d3d893399ff35c39d6fdf4e017fbf066a41d1 14f2243ec33e5a224ba68894c73470159309941d627fc203791695cb70ef228b c63c5cce71c9211ea0b64264d3a19c176fed61d835c125be75786bbc769a0205 4df2d0032e0ee88e798543d76ed25d0f90e2c7b7abf08639e1f5355ca1b3d025 5d81b13e65bee3891422d1d5b2d5e31e07f43c81dde0eeeec4e4b1c54786352c f9e578039a50ef83b0d31f260d38a5106157e1f0772606a6f90725d23e6f9d49 292a9ceac0e4e07557360da68d39db22107bcb64c28d7094b652a702c6999134`
F3 `72431f2558bcb7c421fcf8bfedd58f34ea33460331a9bfb8df96474add31e187 1cfae50359f8696ea2f5cd4ef518836e1a0bd746d7973d8631f0833b73df4cfc 80b992e576b9548081613ad8475064e28ed56cfff7956ada61cec77a217fb0df 8063c060413e378ffe1cfcfbdd1af1f57dcef437d3e9a32c834e915254d32125 67e754f6e18273c62f5dbe05ce5fef326dda359731dfa6c8558bd64fac690ed0 3d532ffe714838429c15cbd1dd78ed747976de040418723dcf2c5c654cd43f9d 8878ea093ff6e236fddcc9417a6dc213cb58c25df9ee42bd351e2af9c224767d a6bddfe8a095c1c7ffbfa939b7b86a6b6ebc9ef31e65e8a8de0e5ac1ca3cd115 b8100a5213ff0b8110533b2045fe35d3d3d52581082588dd999374df85003e34 faa9e2800b87ff15306810bbc35235639b152c10e8b412a3603f1506251c59a3`
F4 `c55a06250ab8286bc604cb23915f03568b5b54ca5dc07033efb659b2c755a951 814b3c3ee490389d4f526717611cb759bb1df80fe45975c206b3b342c03f729b 89ad8d17741dd26b9a4917773b887aa9881a5d89337189f754218951e0b54bf5 c60fbf2043dd20873006737e36bcf52489a14fc813457f899c38ca39c269d04b 9cda47655614d257e5c7295bc4c8a573666193bcf2a809ee2e7d82a09426f3da fd532df17c8b3bebccb8f9c423bdafb989a2e5bb16180a138c41580ee615b85b d90574c9a7adcf367b0b160f472236bbdad4c39bd1c3fc87f1301951efddeb9c 536f08f9117e7d22a3b0c9cb16af152520a65bd2577c9d1e84a0fbe41ca57dac f4b690fa6bc6da4e3b47d3b420f8e418e1befb21ea6c897909fd27c964edbc2e f325bd78c4b43fed2572a618899f935a1e8373f24566e7def0ceb8b28172705f`

正式产物：`/tmp/IAFiLM_20261023/report.md`，fold_metrics.csv（40折），oof.csv（3484行），summary.json，audit.json，manifest.json。NO_GO：F3虽有非零表型调制，但未同时改善位置、KL消融、患者对应及四指标；尤其F3相对M3 ACC/SEN下降，F3相对F4 SEN下降，故不建议将其作为已成立的MICCAI 2027创新点。此seed911已参与设计；未来需冻结方法后以未参与设计的seed、无泄漏RFE和独立站点验证确认。不得在本实验基础上擅自增组或调参。

handoff前缀字节数=218804，前缀SHA256=`fd2ae885fe9af470fcb1c78f5503d3435e3a8ee9ae8cba98ccc879f9940dccff`；追加后应保持该前缀原样，结束标记只出现一次。

IAFiLM_20261023_END
