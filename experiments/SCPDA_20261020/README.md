# SCPDA_20261020

本轮在 /tmp/SCPDA_20261020 执行，原因是用户报告 /data 再次出现 I/O 故障。历史输入仍只读，不复制大体积数据。全部完成后尝试一次幂等追加根 handoff；若写入失败，本目录保留 handoff_pending.md 和恢复命令。

## 固定六组

E0 原生 MM-GTUNets；E1 实验12 D4；E2 D4+EMA；E3 D4+类别独立 ESS 收缩；E4 二者组合；E5 E4 仅辅助目标使用训练内同站点年龄/性别联合错配。仅 seed911，原始10折，共60次完整训练；不额外冻结推理或搜索。

D4 的 prepare、私有 SHA256(seed/fold/PCDCA/purpose)、3子折及错配完全保留。原 beta=clip((minESS-8)/32,0,1)。新 beta_c=ESS_c/(ESS_c+32)，无局部支持回退对应全局类中心。EMA 在 epoch0 初始化当前 train 影像输出，之后 M=.9*M+.1*detach(z)，每训练epoch更新一次。以 EMA 统计同时标准化当前 query/teacher，tau 沿用 D4 公式。所有 teacher/核/原型/ESS/beta/坐标统计均 detach；只有 query 反传。无新参数。

## 原生协议

精确读取 DRPA seed911 已验收的 splits、RFE500、train_start/core、未注册RP权重、完整RNG及VAE路径。一次原始训练forward，一次原始验证forward；val CE+Graph 严格最优选模，EarlyStopping 相等损失重置，最多500epoch/patience100；Adam lr1e-4/wd5e-4。正式测试用最佳权重、训练末RNG、grad-enabled单次随机前向。ASD=0、TD=1，AUC=softmax第0列；十折SD ddof=1。

保留原生随机VAE、eval Dropout、共享RP卷积.data投影、未注册RP权重、RP循环内提前返回、torch.tensor梯度截断、全节点BN，以及历史跨折RFE和转导图局限。原模型仅有既有PPU 1x1投影替换。本轮不作临床泛化或显著性宣称。

## 验证与运行

独立代码以实验12源码作为只读参考，不在正式训练中 import 历史实验。仅预检编译历史源码作为 D4 对照。两锚点各3个完整训练/验证步比较官方AST/E0和历史D4/E1，核对logits/损失/teacher/kernel/ESS/beta/梯度/Adam/BN/TopK/RNG；另验证EMA、空支持、类别独立beta、heldout边界、参数数量及错配隔离。

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/venvs/mm/bin/python -B -u /tmp/SCPDA_20261020/launch.py --stage all
```

阶段可选 preflight、baseline、d4、methods、all。双卡 fold%2 固定；每卡最多4worker，每worker1CPU线程，实时显存/cgroup RAM/CPU限流，OOM降并发不改算法。完成文件先验内容SHA/配置输入/最佳core/指标后跳过；任务锁、原子保存和读回验证，未完成从原始锚点重启。不覆盖健康结果。

最终 analyze.py --final 验收60折、输出 fold_metrics.csv/oof.csv/summary.json/report.md，再追加handoff。若最终写入失败，analyze.py --append-handoff 只重试交接，避免重训。manifest.json 保存保护SHA、锚点、预检和调度记录。原型仅前3及每25epoch统计，完整损失和动态梯度比例按epoch记录，无大型调试图或表示缓存。

## 审计依据

[原论文](https://arxiv.org/html/2406.14455v3)的 MRRL/RFE/VAE、AMRS、双GTUNet和ACMGL联合贡献不等于独立正确条件增益；本轮以本地历史实际路径为基线。[官方代码](https://github.com/NZWANG/MM-GTUNets)固定 bb9351b931003535503694ad2f777b6cee1c151f。已阅读根源码8文件、handoff、实验12配置/prototype/run/launch/analyze/report/summary、DRPA初始化锚点与原始PPU复现记录。历史seed911 D4与D2的ACC近似相等，AUC仅小幅改善；三seed D4整体未建立稳定创新证据。不能事前断言新方法有效。

不自动提交或推送 Git。

## 正式基线重放的漂移核查

E0 新鲜十折 ACC/AUC 为 82.4464/88.4182%，相对历史 D0 为 +0.3422/+0.3170pp，8/10 折最佳 epoch 相同。E1 新鲜十折为 81.8717/88.1644%，相对历史 D4 为 -0.6870/-0.5253pp，超过预先固定的 0.5pp 描述性漂移提醒；没有据此重跑或修改配置。

E1 与历史 D4 的10折起点指纹均相同，9/10折最佳epoch、总epoch和正式测试前RNG相同。8折ACC相同；fold0 ACC差为-2.2727pp，fold2为-4.5977pp。fold2最佳epoch从175变为152、训练总epoch从276变为253，AUC差为-4.5069pp，是总体AUC差的主要来源。最初3步原生训练损失的生产运行差异最大约7.81e-6；两代表性锚点在独立确定性诊断中的输出、原型、梯度、Adam、BN、TopK、RNG仍严格一致。

这些事实支持原生PPU数值和长轨迹/选模漂移需要纳入解释，不能据此证明唯一原因，更不能宣称完整训练与历史逐位复现。主比较使用本轮E1，但E4相对该较低E1的收益不能脱离E0、历史D4和错配对照独立夸大。完整逐折只读核查保存在run.log的historical_D4_fold_drift_audit事件和manifest的historical_replay_checks；未增加模型前向或测试择优。

