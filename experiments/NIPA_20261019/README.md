# Experiment 13: NIPA

独立三因素原生路径归因，不提出新模型、不新增参数/Loss、不自动Git提交或push。

## 审计和边界

已逐项阅读原始8份源码、完整handoff及实验12的README/prototype/run/launch/analyze/finalize/report/summary。原论文、官方与当前GitHub另行核对，实际输入来自DRPA保存的context、splits、RFE500、VAE和train_start。原论文[MM-GTUNets](https://arxiv.org/html/2406.14455v3)Table5报告ABIDE影像/联合ACC76.49/82.92%，但该比较没有分离图、编码及融合，不能当作正确患者对应关系的独立证据。[官方源码](https://github.com/NZWANG/MM-GTUNets)；[当前仓库](https://github.com/tangaii/MM)。

源码VAE输入SITE/SEX/AGE、输出500维且每forward重载并采样。影像RFE500与VAE500拼接后，实际Torch分支使用Euclidean cdist/全队列平均sigma而非论文的correlation距离。RP构图在SITE循环内提前return；SEX/AGE零通道经sigmoid贡献常量背景0.5*(wSEX+wAGE)。训练reward/penalty含训练标签，所有heldout相关motivation不筛站点。原生共享卷积.data约束、未注册RP权重、torch.tensor重建detach、VAE/eval Dropout及BN全部保留；只有既有CustomConv2d末行PPU兼容替换，不在本轮修改。

双GTUNet各输出871x16受试者表示，注意力融合两分支及算术均值，Classifier为16->256/ReLU/BN->2。原图正则使用真实嵌入和原生模态权重。原始RFE跨折监督泄漏、871节点转导图均保留；本轮不是无泄漏临床泛化或未见站点验证。

历史DRPA未建立可靠匹配；PCCI残差退化；PCND无稳定条件增量；LCRA局部对齐与极弱图推理效应不同；CMCV轻量探针无稳定AUC/CE增量但不证明任何深度互补不可能；GBSA身份文本匹配未成功；PCDCA有弱条件迹象但无稳定全局原型优势。因此只拆原生路径，不加Gate/对齐/语义模块。

## 固定因素

G仅在图相似度输入使用ph_z[perm_G]，真实ph_z仍进入ph_unet。F先在真实两路嵌入上计算原生图正则权重，再以第二次无随机Attention计算img/ph[perm_F]分类联合表示；Classifier仍一次。R仅将SITE置换后传入原生RP builder，真实训练标签和其余输入不变。G/F按分区x真站点独立循环错位，单例不动；R按分区标签盲最大化站点改变、频数不变。三套私有SHA种子、映射及有效比例保存audit及每fold。

N0全real且直接调用原core；N1 G-null；N2 F-null；N3 G/F-null；N4 R-null；N5 G/R-null；N6 F/R-null；N7全null。所有组同容量、初始化、Adam顺序、历史训练起点及RNG。Null不是移除非影像；N7不是纯影像。图干预会改变两条GTUNet和Graph Loss，不是单边独立因果效应；RP收益可能包含监督关系协议，不等于生物学互补。

## 执行

使用/root/venvs/mm/bin/python，全部神经计算仅PPU。每worker1CPU线程，每卡最多4worker，动态GPU/cgroup RAM/CPU限流，fold%2配对物理卡；不强杀健康任务。

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/venvs/mm/bin/python -B -u experiments/NIPA_20261019/launch.py --stage all
```

可分阶段preflight/baseline/methods/frozen/analyze；先5锚点官方AST/N0/all_real连续3步严格等价和全8因素隔离，再N0 30折并核查C0及D0的0.5pp漂移，随后N1-N7交错210折。最后30个N0最佳Checkpoint、3预注册历史RNG定义draw、8条件共720完整冻结前向；逐条件恢复core/BN/未注册RP/RNG，无backward或optimizer。只在全部研究和文件验收完成后幂等安全追加handoff。

每fold只保存一个SHA封装产物runs/Nx/s{seed}_f{fold}.pt；N0冻结结果合并回其产物但正式字段指纹不变。完成任务验证后跳过，未完成从历史起点重新开始，唯一任务锁、fsync/安全发布/关闭重读，仅暂时ENOENT有限重试，EIO不吞掉。没有VAE/RFE训练，没有DDP。测试使用ASD=0连续概率；每seed先10折平均，再三seed均值与样本SD。正式重训差异和同Checkpoint直接敏感性分别报告，不以30相关fold当独立队列。
