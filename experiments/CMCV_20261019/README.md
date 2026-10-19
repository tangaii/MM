# Experiment 10: CMCV

本轮只训练固定Logistic Regression与RBF-SVM，不训练MM-GTUNets、GTUNet、VAE或新增神经网络。用户因`/data` I/O故障授权全部新实验在`/tmp/CMCV_20261019`完成；历史项目与实验只读。

Protocol H只引用历史成绩，不与本轮探针绝对分数作方法排名。Protocol R保留历史30个train/val/test成员，但每折从6670维FC独立拟合训练集RFE500及预处理。Protocol S为20站点LOSO，无站点编码输入。每个任务共用一次RFE。

固定参数、块尺度、缺失/未知处理、8个分区内同站点联合错配及2000次站点聚类Bootstrap见`config.json`。所有P/K组保留统一槽位；P4/P5只置零未启用通道。P3/K2分别拟合8个Null，不将8次结果当作独立数据集或用概率平均替代其主结果。邻居N2a仅错配训练锚点，N2b同时错配查询与锚点，区别保留。

运行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /root/venvs/mm/bin/python -B /tmp/CMCV_20261019/run.py --execute
```

`run.py`依次执行输入/存储审计、集中单测、四个完整技术预检任务、其余正式任务与独立分析。每任务单一`runs/<task>.bin`先原子保存RFE阶段，后替换为完整结果；内置SHA256验证。持久锁只用于互斥，验收后清理无活跃任务的锁文件。完成且源码/输入指纹匹配的任务直接跳过。失败不吞掉、不伪报完成。

轻量sklearn探针使用CPU，这是本轮用户明确授权的范围。两条逻辑设备队列各25个任务并不等于两卡各完成25个GPU训练；实际PPU只用于必要的数值/速度测量，若传输后无收益则正式任务使用CPU，报告会分别列明。

结果：`audit.json`、`summary.json`、中文`report.md`、单个`run.log`及50份任务结果。`inputs.bin`是经过原始FC逐受试者核验的必要本地只读输入缓存，避免反复访问故障FUSE。无自动Git提交或push。全部完成后才尝试锁保护、幂等安全追加根`handoff.md`；失败则保留待追加文本并明确报告。
