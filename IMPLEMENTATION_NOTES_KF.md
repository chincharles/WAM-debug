# KF P0 实施记录

## 基线与仓库检查

原工作目录只有用户任务书和系统 `.DS_Store`，没有既有代码、权重或实验结果。已完整阅读 855 行任务书；未发现适用的 AGENTS.md。在新目录 `SimWAM-KF/` 克隆官方仓库，创建分支 `research/kf-v1`，保留任务书原文件。

- 上游 SHA：`68b426c162827cb7701396895dbb3572d29f3420`。
- NAVSIM：该 SHA 携带的 `navsim/`，包版本 1.1.0，tree `757d502937265778a3891c3d6e95099952fc80cd`。
- 修改尚未提交为 Git commit；`git diff` 查看上游修改，`git status --short` 同时查看新增文件。没有复制第二套上游算法。

## 文件与职责

| 文件/目录 | 实际实现 |
|---|---|
| `models/wan22/future_generator.py` | FuturePacket、独立噪声的冻结 video-only 采样、latent 时空元数据、禁止 RGB 解码 |
| `models/wan22/future_adapter.py` | 空间 patchify、位置/模态标识、门控 cross-attention；参考专家自行持有模块 |
| `models/wan22/simwam_kf.py` | KF 注册、场景/候选展开、F0/F8 共用 velocity 推理、严格 C0/C1 加载、LoRA 合并导出 |
| `runtime_kf.py` | 显式接收/检查 KF 配置，不把 KF 字段透传原构造函数；禁止自动模型下载 |
| `trainer_kf_warmup.py` | 上游 action scheduler 的 velocity flow-matching；只训练适配器 |
| `trainer_kf_grpo.py` | 原 FlowGRPO loss、原采样接口的实际调用；独立随机流、完整 rollout reuse、冻结名单、NCCL 梯度归约、日志/恢复 |
| `datasets/navsim/kf_validation.py` | manifest 限制、只加载当前摄像头、严格官方 PDM reward（异常不转零分） |
| `kf/contracts.py, config.py, execution.py, scoring.py` | 配置/噪声/split/单位契约、Hydra 继承、路径/模型实例化、环境和权重 hash |
| `configs/task/navsim_kf_*.yaml` | 在固定 FlowGRPO task 上继承 warmup/RL 两个任务 |
| `scripts/kf/` | 环境模板、官方 split 导入、preflight、真实 GPU regression、warmup/train/eval、smoke、四组矩阵、机制诊断、固定prompt文本缓存生成和汇总 |
| `tests/kf/` | CPU 缩小版真实 DiT/MoT 单元/循环/CLI 测试；显式 GPU integration marker |
| `README_KF.md` | 中文服务器准备、全部命令、恢复、产物与故障边界 |
| `requirements-kf-cpu.txt, verification/` | CPU 开发依赖、本次测试记录、精确依赖版本；没有实验分数 |

以上 `models/`、`kf/` 等路径相对 `src/simwam/`。

上游最小修改点：

1. `simwam_grpo.py` 返回当前 latent，动作 block 后执行可选适配器，支持显式初始/逐步噪声，校验形状；F0 数学路径不变。
2. `lora.py` 注入时排除 `future_adapters` 子树。
3. `trainer_grpo.py` 允许调用者提供独立随机流预生成的 F0 reference chain；PPO/BC 数学式保持原实现，补 ratio 分位数。原入口未提供该字段时保留原 BC 采样行为。
4. `runtime_grpo.py` 工厂增加可选模型类；KF 使用子类实例而非替换现有对象的类。
5. `simwam.py`、`runtime.py`、helpers/utils 包初始化延迟加载 I/O/训练/视频依赖，CPU tiny imports 不要求整个 GPU/NAVSIM 环境。
6. `helpers/io.py`、`helpers/loader.py` 在 KF 模式禁止隐式下载并审核 VAE 键；非 KF 上游默认行为保留。

## 实施选择与边界

- 完整 C0 提供视频/动作/proprio；构造模型时跳过单独加载 Wan/ActionDiT backbone，再强制装载 C0。不会随机初始化后继续训练。独立 VAE 必须存在且通过加载/hash 审核。
- 视频 microbatch 目前只支持 1；大于 1 会拒绝，避免配置声称使用了未实现的批处理。缓存是每 rollout 原始 detached future latent；所有 inner epochs 复用该对象。没有跨 rollout 缓存。
- latent 时间坐标使用 VAE 时间压缩组末端秒数（F8 为 2/4 秒两个 latent 时间位置），原始 `frame_times_s` 仍保留 0.5～4 秒的 8 帧定义。这不是把 F 当作 latent 时间长度。
- BC reference chain 在每 rollout 预先用独立 reference 随机流生成并复用；BC 仍是上游 reference transition log-prob anchor，F0，系数0.1。它不是显式 KL。
- 训练循环复用上游 `_policy_loss` 而采用新的小型外层循环。每个 rank 完整模型、显式 all-reduce trainable 梯度、无梯度累积、一个 rollout buffer。此选择避免原 unwrapped forward 普通 DDP 同步缺陷；不是 ZeRO 内存优化。配置中相应未支持选项显式报错。
- sampler 用稳定 seed 的全局 epoch permutation，按 rank 取观测，drop-last；统计记录每 rank、world size 和累计场景/候选。没有保证跨 world size 或硬件逐位一致。
- 精确 resume 限于 rollout 边界，同一配置/world size/数据/初始化/max_steps；保存当前 MoT、proprio、reference、优化器、调度器和所有随机状态、数据进度。VAE 固定来源/hash。导出与恢复文件严格区分。
- 独立正式 val 在每组训练结束执行，统一最后 checkpoint；没有按上游 `eval_every` 在线执行正式验证，也没有实现验证集最优选择。旧训练集 eval 明确禁用。
- PDM 直接调用固定 NAVSIM v1 `pdm_score`，官方默认 scorer 单独实例化；所有场景有效才取均值，没有异常零分或 best-of-K。训练 reward 配置不会替换正式 scorer 默认值。
- 原数据集使用固定 odo 常数，忽略 JSON stats 的数值。沿用该行为，并将 stats 文件作为来源/hash 校验项；不会自行重估或修改动作单位。
- GPU-hours 是训练循环 wall time × world size（不含模型装载/preflight），不是等算力控制。rank0 可读日志不当作全局 reward；其他 rank 单独保存。

## 本地已执行验证

实际使用 macOS ARM64、Python 3.12.14、PyTorch 2.7.1 CPU。原系统 Python3.9 低于上游要求，最终使用单独 `.venv-kf312/` 执行；没有把3.9环境结果当成验收。

`verification/kf_cpu_tests.txt` 和 `verification/kf_gpu_tests.txt` 保存最终命令输出；精确 CPU 依赖见 `verification/kf_cpu_freeze.txt`。测试覆盖：

- F0 与原 `SimWAM.infer_action` 输出逐值一致，F0 不调用未来生成器/适配器。
- 原视频专家 tiny video-only 与 joint latent 对齐（rtol1e-5/atol1e-6）。
- 零 gate identity、非零 gate 条件敏感性、全部适配器参数梯度。
- PPO 初始 ratio=1（F0/F8），扰动未来使 log-prob 改变，raw packet 复用。
- B>1/K>1 候选顺序和独立未来，K4 与 K8 前四候选噪声匹配。
- 预热只动适配器；完整4-update CPU RL 循环分别覆盖 F0/F8，只动 LoRA，reference/原权重/适配器不变。
- 合并导出重载 F0/F8 回归、缺适配器键失败、optimizer/scheduler/reference/RNG/step 恢复。
- 常量优势 PPO loss=0、候选复制不使 loss 成倍增大。
- 未来图像/专家轨迹扰动不改变当前策略条件。
- split log 交集、重复 token、无 cache、越界 token、NaN、reward/PDMS 单位错误。
- 四组 Hydra 继承、显式工厂参数、未知 KF 键、F4/oracle/不同正式步数拒绝。
- CLI help、四组完整 dry-run及8个 eval 计划、非空目录保护、缺资源 preflight。

另执行 compileall、全部 KF shell 脚本 `bash -n`、`git diff --check`。测试夹具中的合成奖励不是 NAVSIM 分数；没有输出到正式研究汇总。

## 尚未执行的真实检查

原因：本机没有 NVIDIA GPU、真实 C0/C1、VAE 权重、NAVSIM 日志/传感器/地图/文本和 metric cache。

1. 完整5B权重的严格加载与结构匹配、真实VAE编码/latent形状。
2. 真实 C0 F0 回归及 video-only/joint bf16 latent 容差实测。
3. 真场景 warmup backward、四组 RL update、有限 reward/梯度、门控与敏感性检查。
4. C0/C1 F0 真轨迹回归、导出模型 F0/F8 官方评分。
5. GPU allocated/reserved峰值、实际延迟、GPU-hours、1/2/4/8 GPU稳定性与跨进程恢复。
6. 四组 pilot、三 seed confirm、最终 test、交互/置信区间/机制分析。
7. Linux GPU完整依赖安装与 `pip check`。保留服务器真实验证后的 lock，而非宣称本地验证了 CUDA 环境。

这些外部检查已经有脚本入口，但没有任何“已跑通真实模型”或“提高 PDMS”的结论。P1/P2（F4/K16/联合适配器RL/共享未来/动态K-F/环境分叉）明确未实施。
