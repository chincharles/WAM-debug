# SimWAM-KF P0 运行说明

本目录是实际实现的 P0 研究原型。CPU tiny 测试通过不代表真实模型或实验通过；当前没有 NVIDIA GPU、C0/C1 权重或 NAVSIM 数据，没有训练分数、PDMS 或硬件测量结果。真实运行前必须执行下面的 preflight 和 smoke。

上游固定为 `68b426c162827cb7701396895dbb3572d29f3420`，工作分支 `research/kf-v1`。复用本提交携带的 `navsim/`（包版本 1.1.0，NAVSIM v1 协议），其 Git tree 为 `757d502937265778a3891c3d6e95099952fc80cd`。不要安装同仓库的 `navsim_v2/` 覆盖它。

## 已实现的链路

- `K=4/8` 的唯一来源是 `grpo.sample.group_size`，`F_train=0/8` 来自 `kf.future_frames`，`F_eval=0/8` 来自 `evaluation.future_frames`。动作固定 8×3，4 秒，正式评测固定单候选、10 步 ODE。
- F0 保留原始当前图像/VAE/KV 路径，直接 bypass FutureAdapter。F8 由冻结的视频专家从当前帧生成 9 帧序列对应的 latent，固定首帧，只留下未来 latent；默认 20 步，完全不解码 RGB。
- 每个候选有独立 future seed；逐候选 microbatch=1。未来条件不是动作条件反事实，不使用真实未来图像或专家动作作为策略输入。
- FutureAdapter 注册在动作专家的 `future_adapters` 下，在 block 9/19/29 后执行；2×2 patch，256 维，8 heads，时空位置与模态编码，零初始化标量 gate，非零输出投影。GRPO 和部署共用动作 velocity/block runner。
- C0 → 仅适配器 flow-matching 预热 → C1。四组从同一 C1 出发，只更新原动作注意力 LoRA（r16、alpha32）。F0 BC anchor 使用冻结 C1 参考动作专家。
- 复用上游 SDE、scheduler、transition log-prob、PPO clip、denoising discount 和 candidate mean；不是用 MSE 代替 PPO。`upstream_compat` 的裁剪/平均 log-prob 不是严格联合轨迹密度。
- manifest 限定当前图像读取和评分 token；train/val 按 log 分离；正式评分调用 vendored `navsim.evaluate.pdm_score` 与默认官方 scorer，训练 reward 为 0～1，报告 PDMS 为 0～100。

## 迁移代码

可以复制本项目代码目录，保留 Git 信息并排除本地虚拟环境、outputs、runs 和数据；也可使用交付的 `SimWAM-KF-P0-overlay.tar.gz`：

```bash
git clone https://github.com/H-EmbodVis/SimWAM.git SimWAM-KF
cd SimWAM-KF
git checkout -b research/kf-v1 68b426c162827cb7701396895dbb3572d29f3420
tar -xzf /path/to/SimWAM-KF-P0-overlay.tar.gz
```

overlay 只包含本扩展新增/修改的代码、配置、文档和本地验证记录，不含模型、数据、虚拟环境或实验结果。只应用于新克隆的上述版本。`verification/delivery_manifest.json` 列出文件及 SHA256。

## 环境安装

### 本地 CPU 开发

使用 Python 3.12（本次实际为 3.12.14）。不下载模型或数据：

```bash
python3.12 -m venv .venv-kf
source .venv-kf/bin/activate
python -m pip install -r requirements-kf-cpu.txt
export PYTHONPATH="$PWD/src:$PWD/navsim:${PYTHONPATH:-}"
python -m pytest tests/kf -m "not gpu and not integration" -q
```

本次实际环境位于 `.venv-kf312/`，精确依赖记录为 `verification/kf_cpu_freeze.txt`。tiny tests 使用真实上游缩小版 DiT/MoT/scheduler；VAE 编码、数据集和奖励仅在测试夹具中缩小/替换，不产生研究结果。CPU 环境与 GPU 完整环境分开，避免 CPU NumPy 版本影响 NAVSIM 旧依赖。

### 与现有 vla 服务器匹配的 PPU 环境（优先）

已核对 vla 的 `scripts/navsim/ppu_environment.py`、`requirements-ppu.txt` 和环境记录：服务器是 **PPU-ZW810E，Linux / Python 3.12，厂商 torch 2.6.0 / torchvision 0.21.0**。设备通过厂商 `torch.cuda` 和 `nccl` 兼容接口暴露。不能直接安装本仓库根目录的 `requirements.txt`，其中 torch 2.7.1、torchvision 0.22.1、Triton 和 DeepSpeed 会与该镜像冲突。也不要激活 VLA 的训练虚拟环境来运行 SimWAM。

新增 `install_ppu.py` 基于 vla 的隔离方式：在独立 venv 链接镜像中的厂商框架文件，保留 SDK/动态库环境；约束版本、审查 pip dry-run 安装计划，拒绝替换加速器包，再安装审查过的下载产物。排除无关 DALI 包，最后执行 pip check、框架路径核对和实际模块导入。SimWAM 的 transformers/accelerate 等使用本项目版本，不照搬 VLA 的旧版本。

此配置沿用 vla 的 **nuPlan 1.2.0 固定提交 `ce3c323af01c0d7ec5672f7832ef53f9c679aab0`**，以外部 build 目录构建，规避源码中同名 `build` 文件。与本项目原始 nuPlan 1.1.1 pin 不同，属于显式服务器适配；NAVSIM 仍使用本仓库 vendored v1，绝不引入 vla 的 NAVSIM 或 v2。`np.int = int` 兼容别名只在 PPU 环境启用，不修改评分公式。所有四组实验必须使用同一套环境，正式结果前仍需真实 metric cache / 官方评分检查。

```bash
# 在服务器的 SimWAM-KF 仓库根目录，使用原始 PPU 镜像 Python（不是 VLA venv）。
python scripts/kf/install_ppu.py \
  --venv "$PWD/.venv-kf-ppu" --output "$PWD/reports/server-setup" --devices 2
source .venv-kf-ppu/bin/activate
cp -n scripts/kf/env.aliyun.sh scripts/kf/env.local.sh
source scripts/kf/env.local.sh
source scripts/kf/env.ppu.sh
python scripts/kf/check_server.py --output reports/server-imports.json
# 合成算子测试：不需要数据和权重，不是模型实验。
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/kf/check_server.py --kernels --output reports/server-kernels.json
python -m pip freeze > reports/environment-server.lock.txt
```

`--devices` 是最少可见设备数；实际单卡服务器改成 1，同时将 kernel 命令的 `--nproc_per_node` 改成 1。检查涵盖 bf16 masked SDPA/backward、checkpoint、真实 RoPE 路径、Conv3D/AdamW、设备 RNG 恢复和多卡 all-reduce。它不能证明完整 Wan/VAE 已兼容或能放入显存。**本地没有 PPU，安装和上述算子检查尚未在目标服务器执行。**

### 普通 NVIDIA 服务器（仅非 PPU）

上游环境是 torch 2.7.1 / torchvision 0.22.1 / nuPlan 1.1.1，必须另建环境，按驱动选择 CUDA wheel。不要将 PPU 安装流程与根目录 `requirements.txt` 混用。此前通用 Python 3.10 安装示例不适用于已记录的 PPU 镜像；当前主要维护上述 Python 3.12 PPU 配置。

P0 启动器接受 `NPROC_PER_NODE=1/2/4/8`，每个进程持有完整模型，以显式 NCCL all-reduce 平均可训练参数梯度。没有使用 ZeRO 分片；增加 GPU 数不会让一个过大的模型自动装进单卡。这里选择显式梯度归约，是为了避免上游 unwrapped forward 在普通 DDP 下不同步的问题。GPU 数、global batch 和 seed 计划须在四组中固定。默认每 rank 1 个观测；不足 global batch 的尾部丢弃并在下一 epoch 重新排列。没有自动降 K/F/分辨率来应对 OOM。

## 服务器资源与路径

先复制、编辑并 source 环境模板：

```bash
# 已迁移 vla 项目提供的阿里云 CPFS 数据路径；其他服务器可用 env.example.sh
cp -n scripts/kf/env.aliyun.sh scripts/kf/env.local.sh
# 数据根目录已填写；将 env.local.sh 中剩余 /path/to（权重、缓存、统计）替换为实际路径
source scripts/kf/env.local.sh
```

| 资源 | 配置变量/路径 | 要求 |
|---|---|---|
| OpenScene v1.1 trainval 日志与传感器 | `NAVSIM_LOG_PATH`、`NAVSIM_SENSOR_BLOBS_PATH` | 使用 NAVSIM v1 场景过滤器；仅读取当前前视摄像头 |
| 最终 test 日志与传感器 | `NAVSIM_TEST_LOG_PATH`、`NAVSIM_TEST_SENSOR_BLOBS_PATH` | 只在明确 `--split test` 时评分 |
| nuPlan maps | `NUPLAN_MAPS_ROOT`、`NUPLAN_MAP_VERSION=nuplan-maps-v1.0` | 与日志的地图位置匹配 |
| train / val / test metric cache | `NAVSIM_METRIC_CACHE_PATH`、`NAVSIM_VAL_METRIC_CACHE_PATH`、`NAVSIM_TEST_METRIC_CACHE_PATH` | vendored v1 生成；包含 metadata 索引；选定 token 100% 覆盖且可解压 |
| 文本 embedding 缓存 | `NAVSIM_TEXT_EMBED_CACHE` | 上游固定 prompt、context_len=256、4096 维 umT5 缓存；缺失时用新增 `python scripts/kf/cache_text.py --help` 准备 |
| 统计 JSON | `NAVSIM_STATS_PATH` | 保存并校验来源/hash。注意：此上游实际上使用固定 odo 边界，统计文件不参与重算归一化 |
| 已训练的完整 SimWAM IL C0 | `SIMWAM_IL_CHECKPOINT` | 必须包含 `mot`（视频与动作专家）和 `proprio_encoder`，不能仅给 Wan 或 ActionDiT backbone |
| VAE | `$DIFFSYNTH_MODEL_BASE_PATH/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors` | 上游重定向的 Wan2.2 VAE；加载键审核与 hash 校验 |
| 共同 C1 | `SIMWAM_KF_CHECKPOINT` | warmup 产出的 `export/kf_policy.pt`；预热阶段不要求已存在 |
| manifests | `KF_TRAIN_MANIFEST`、`KF_VAL_MANIFEST`、`KF_TEST_MANIFEST` | JSONL，每行至少 `scene_token`、`log_id` |

缓存准备的可选命令（只在资源缺失时运行；不会下载权重）：

```bash
# navtrain cache 可同时覆盖 train/val，访问范围仍由两个 manifest 分别限制
TRAIN_TEST_SPLIT=navtrain CACHE_PATH="$NAVSIM_METRIC_CACHE_PATH" \
  bash navsim/scripts/evaluation/run_metric_caching_train.sh
export NAVSIM_VAL_METRIC_CACHE_PATH="$NAVSIM_METRIC_CACHE_PATH"
TRAIN_TEST_SPLIT=navtest CACHE_PATH="$NAVSIM_TEST_METRIC_CACHE_PATH" \
  bash navsim/scripts/evaluation/run_metric_caching.sh

# 若没有兼容文本缓存，额外准备本地 umT5 权重与 tokenizer，然后运行
python scripts/kf/cache_text.py \
  --weights /path/to/models_t5_umt5-xxl-enc-bf16.safetensors \
  --tokenizer /path/to/google/umt5-xxl \
  --output-dir "$NAVSIM_TEXT_EMBED_CACHE"
```

上游报错文本曾引用未随此提交提供的 `cache_prompt_embeddings` 模块；本扩展提供可执行的 `cache_text.py`，复用上游固定 prompt 与 `encode_prompt`。该可选大文本编码器链路尚未在本机执行。

因为 C0 必须完整，KF 任务设置 `skip_dit_load_from_pretrain=true`，先构造结构，再严格装载完整 C0；不再要求独立 ActionDiT 初始化权重或重复加载 Wan 视频初始权重。缺少 C0 立即失败，不会以随机动作模型继续训练。VAE 仍须单独准备；使用缓存文本时无需在执行环境加载 T5 权重。所有 KF 入口禁止自动下载模型。

`prepare_splits.py` 先保留官方 train/val log 定义；若 navtrain 中没有官方 val log，才从官方训练 logs 按 seed 做 log holdout，并记录方法。navtest 独立生成。已有 manifest 也可直接设置环境变量使用，preflight 会检查 token 源、log ID、交集和 cache。可使用固定训练 subset，但四组必须完全一致。

## 完整命令链

从项目根目录执行；已有非空实验目录默认拒绝覆盖。

```bash
source scripts/kf/env.local.sh
python -m pytest tests/kf -m "not gpu and not integration" -q

python scripts/kf/prepare_splits.py \
  --source-split navtrain --val-log-fraction 0.1 --seed 2026 --output-dir manifests

python scripts/kf/preflight.py \
  --stage warmup --output outputs/kf/preflight_warmup.json

bash scripts/kf/smoke.sh --output-root runs/kf/smoke

bash scripts/kf/warmup.sh \
  --seed 42 --max-steps 1000 --run-dir runs/kf/warmup
export SIMWAM_KF_CHECKPOINT="$PWD/runs/kf/warmup/export/kf_policy.pt"

python scripts/kf/preflight.py \
  --stage rl --output outputs/kf/preflight_rl.json

bash scripts/kf/run_matrix.sh \
  --stage pilot --seeds 42 --max-steps 200 --output-root runs/kf/pilot

python scripts/kf/summarize.py \
  --runs-root runs/kf/pilot --output-dir reports/kf/pilot
```

smoke 从现有 train/val manifest 各取最多 8 个场景，保留两集合 log 不重叠；只在独立 smoke 子目录写入子 manifest。先比较真实 C0 的 F0 新旧路径、相同初始视频 latent 下的 video-only/joint 结果，再执行 2 步预热、四组各 2 次 RL 更新（inner_epochs=1），以及各自 F0/F8 评分。正式 warmup 自动生成 C0/F0、C1/F0、C1/F8 验证结果，检查 C0/C1 F0 轨迹逐值一致，并对 C1 的生成未来做打乱/关闭诊断。

正式四组默认每个 rollout 做 4 次 optimizer update，因此 200 updates 为 50 rollouts（每 rank）。max_steps 必须是 inner_epochs 的倍数；不在未消费完的 rollout 中途导出“精确恢复”状态。矩阵使用同一 C1，每组结束自动运行 val F0/F8；任何失败立即停止。`train_metrics.rank*.jsonl` 保存各 rank 记录，`train_metrics.jsonl` 为 rank0 可读日志，不应将 rank0 reward 当成全局正式 PDMS。

单组和独立评测：

```bash
bash scripts/kf/train.sh --k 8 --future-frames 8 --seed 42 --max-steps 200 \
  --run-dir runs/kf/single_k8_f8_seed42

python scripts/kf/eval.py \
  --checkpoint runs/kf/single_k8_f8_seed42/export/kf_policy.pt \
  --split val --future-frames 0 --candidate-count 1 --action-steps 10 --seed 2026 \
  --output-dir runs/kf/single_k8_f8_seed42/eval_val_f0

python scripts/kf/eval.py \
  --checkpoint runs/kf/single_k8_f8_seed42/export/kf_policy.pt \
  --split val --future-frames 8 --candidate-count 1 --action-steps 10 --seed 2026 \
  --output-dir runs/kf/single_k8_f8_seed42/eval_val_f8 --diagnostics

bash scripts/kf/run_matrix.sh --stage confirm --seeds 42,43,44 --max-steps 2000 \
  --output-root runs/kf/confirm
```

最终 test 使用同样 evaluator，显式改 `--split test` 和独立输出目录；默认矩阵不会使用 test。先确定最终检查点/协议，再执行 test。

`--help` 和 `--dry-run` 不需要权重。dry-run 输出可解析的 resolved config/计划，不声称执行成功。例如：

```bash
bash scripts/kf/train.sh --k 8 --future-frames 8 --max-steps 200 \
  --run-dir outputs/kf/plan_k8_f8 --dry-run
bash scripts/kf/smoke.sh --output-root outputs/kf/smoke_plan --dry-run
```

精确恢复仅支持同一 world size、配置、C1、manifest 和 max_steps 的 rollout 边界：

```bash
bash scripts/kf/train.sh --k 8 --future-frames 8 --seed 42 --max-steps 200 \
  --run-dir runs/kf/single_k8_f8_seed42 \
  --resume runs/kf/single_k8_f8_seed42/checkpoints/step_000040
```

每个 rank 恢复当前 MoT/LoRA、冻结 reference、optimizer、scheduler、Python/NumPy/Torch/CUDA RNG、epoch/batch/rollout/update 进度；不可把 `export/kf_policy.pt` 当作 resume。冻结 proprio/VAE 从经 hash 校验的共同初始化重新加载。部署导出合并 LoRA，保留适配器和配置元数据。

## 产物与解释

```text
run/
  resolved_config.yaml
  environment.json             # 源码、git、依赖、GPU、路径/权重 hash
  data_manifest_meta.json
  trainable_parameters.json
  train_metrics.jsonl           # rank0；非正式 PDMS
  train_metrics.rank0.jsonl     # 每个 rank 单独保存
  checkpoints/step_000040/rank0.pt
  export/kf_policy.pt
  status.json
  eval_val_f0/{per_scene.jsonl,summary.json,status.json}
  eval_val_f8/{per_scene.jsonl,summary.json,status.json}
```

场景结果保存 pose、scene/log、训练 K/F、评测 F、train_seed/eval_seed、checkpoint hash、PDMS/NC/DAC/EP/TTC/comfort。PDMS 为0～100，分项保留官方0～1数值。汇总产生 CSV/JSON，保留缺失值；单 seed 的跨 seed 标准差为 null。交互与 F8−F0 推理收益按同一场景配对，置信区间按 log cluster bootstrap（2000 次），跨 seed 另报均值/样本标准差。相邻帧不视为独立 bootstrap 单位。GPU-hours 来自训练日志；延迟在一次预热后同步 CUDA 计时，含当前编码/未来生成，不含数据 I/O、文本编码或 RGB 解码。

## 已知限制与服务器验收

详见 `IMPLEMENTATION_NOTES_KF.md`。当前 CPU 验证不包含真实 VAE、权重、地图/缓存反序列化、真实 PDMS、bf16 数值容差或多卡。GPU regression 的 bf16 默认 atol/rtol=0.02 是待实测审核的起点，脚本会保存实际最大误差；不能仅凭默认阈值证明数值等价。首次运行需审查显存、轨迹单位/范围、gate 与敏感性、ratio≈1、有限 loss/reward/梯度，以及 C0/C1 F0 回归。

正式验证目前在每组训练结束后执行，以最后统一步数检查点为比较对象。没有实现在线验证集最优 checkpoint 选择。完整 preflight 会逐个读取选中场景当前图像/文本并解压 cache，可能耗时较长。全量模型每卡驻留，动作候选不分块，F8 仍可能 OOM；不会自动改实验设置。

P1/P2（F4、K16、适配器联合 RL、共享未来、自适应调度、连续环境分叉）没有实现，不在本 P0 中冒充完成。两档 F 对照不能证明最佳帧数，四组代码可运行也不能证明研究假设成立。

### 从 vla 项目迁移的数据路径

`scripts/kf/env.aliyun.sh` 保存原项目 `configs/navsim/aliyun_paths.sh` 中用户提供的目录：根目录 `/mnt/cpfs-wlc-rdma-300t/navsim/openscene-v1.1`，地图子目录 `map`，日志 `navsim_logs/{trainval,test,mini}`，传感器 `sensor_blobs/{trainval,test,mini}`。训练和测试变量已转换为本项目使用的 `NAVSIM_*` 名称，mini 变量仅供手动选择。`NAVSIM_DEVKIT_ROOT` 仍指向本项目的 NAVSIM v1。

本地 `env.local.sh` 已初始化并被 Git 忽略。上述是服务器路径迁移，没有在本地验证目录存在或数据完整性；服务器仍需执行 preflight。VLA 的模型、预处理产物和 metric cache 未认定与本项目兼容，因此相关路径继续保留占位符，需按本文准备。

### 显式下载基础模型（自己训练 C0）

下载入口 `scripts/kf/download_models.py` 无需 torch，可在联网下载机执行。使用 Hugging Face 的 snapshot_download，第一次解析并锁定完整仓库 revision，重跑沿用锁并续传。会检查视频 DiT index 中全部分片、VAE、文本编码器、tokenizer 文件，写入 SHA256 清单；离线 `--verify` 检查文件是否变化。检查文件完整性不等于模型语义兼容，后续仍需真实加载。勿同时启动多个下载进程写同一目录。

```bash
# 在安装好的环境、仓库根目录执行；此命令为本次准备选择独立路径。
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/cpfs-wlc-rdma-300t/navsim/simwam-kf/models
export NAVSIM_TEXT_EMBED_CACHE=/mnt/cpfs-wlc-rdma-300t/navsim/simwam-kf/text-cache
# 将以上两项同时写入 env.local.sh，后续重新 source 时才会保留。
python scripts/kf/download_models.py --root "$DIFFSYNTH_MODEL_BASE_PATH" --plan
python scripts/kf/download_models.py --root "$DIFFSYNTH_MODEL_BASE_PATH"
python scripts/kf/download_models.py --root "$DIFFSYNTH_MODEL_BASE_PATH" --verify
python scripts/kf/cache_text.py \
  --weights "$DIFFSYNTH_MODEL_BASE_PATH/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors" \
  --tokenizer "$DIFFSYNTH_MODEL_BASE_PATH/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" \
  --output-dir "$NAVSIM_TEXT_EMBED_CACHE"
# 离线转换 Wan → ActionDiT 初始化文件；需要实际设备和足够内存。
SIMWAM_KF_OFFLINE=1 bash scripts/model_prepare.sh
```

下载源为 `Wan-AI/Wan2.2-TI2V-5B` 的视频 DiT、`DiffSynth-Studio/Wan-Series-Converted-Safetensors` 的 VAE/umT5，以及上游 loader 指定的 `Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl` tokenizer 子目录；不会下载整个 Wan2.1 视频模型。默认需要下载数十 GB，另需缓存和初始化转换的存储/内存余量。下载进程显式解除 `HF_HUB_OFFLINE`，不修改父 shell；训练仍离线。可在另一联网机器下载并连同 `simwam-models.lock.json` 整体传到服务器。若使用自有 HF endpoint，在执行下载前设置 `HF_ENDPOINT`。未在本地下载这些大权重。

官方接口说明：https://huggingface.co/docs/huggingface_hub/v0.29.2/en/guides/download 。基础模型文件目录：https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/main 。

### 自训 C0 与 PPU 的剩余边界

下载基础权重后仍需完成 NAVSIM 联合监督训练才能得到 C0；下载和 ActionDiT 转换本身不会产出已训练的驾驶策略。上游 `train_navsim_zero1_torchrun.sh` 使用 DeepSpeed，**不可直接在此 PPU 环境执行**。新增 PPU 环境不安装通用 DeepSpeed。单卡可通过上游 `python scripts/train.py task=navsim_uncond_front_384x672_1e-4 max_steps=1 batch_size=1 num_workers=0 model.mot_checkpoint_mixed_attn=true` 探测监督训练链路，但仍需完整训练资源，且未经真实 PPU 验证；上游默认 val 复用 train，不应将该输出视为独立验证成绩。自训 C0 的多卡 PPU 训练同步、显存方案和独立验证集仍需补充验证，不能把上游脚本当作已适配成功。当前 KF warmup/RL 自身使用显式 all-reduce，不依赖 DeepSpeed。

本次下载源验证状态：已通过网页核对官方 Wan DiT 分片和 tokenizer 目录，转换版仓库采用上游 loader 指定的源；尝试读取其 HF API 清单时网络超时，尚未确认该源在目标服务器可达。下载脚本会在下载大文件之前解析全部仓库 revision，源不可达时失败并保留错误，不会静默替换其他模型。

## 一键分阶段 debug（建议服务器从这里开始）

入口是 `bash scripts/kf/debug_server.sh`：自动使用已经安装的 `.venv-kf-ppu`，读取 `env.local.sh`，启用 PPU 兼容配置；不会自动安装依赖或开始长训练。未安装环境时使用当前 Python，第一阶段会报告缺失依赖。Python 编排器本身只依赖标准库；每阶段通过独立子进程运行，不因缺少 torch 而连诊断报告也无法生成。重试脚本固定本次 Python 和资源路径，不会误激活 VLA 环境。

```bash
# 1. 无需 GPU/模型，先看完整命令计划，不执行任何检查或训练
bash scripts/kf/debug_server.sh --preset full --plan

# 2. 逐步准备；任何一步失败都会停止。先完成环境安装（见前文）。
bash scripts/kf/debug_server.sh --steps environment --devices 2
# 显式选择 download 才会联网下载；先填写 env.local.sh 的模型根目录
bash scripts/kf/debug_server.sh --steps download,models --model-kind base
# 默认 check = 环境导入 → 合成算子/通信 → 基础模型结构检查
bash scripts/kf/debug_server.sh --preset check --devices 2

# 3. 已有自己训练或兼容的 C0，以及数据/缓存/manifest 后：
bash scripts/kf/debug_server.sh --preset smoke --model-kind c0 --devices 2

# 4. 显式启动：环境 → 算子 → C0/VAE → preflight → smoke →
#    1000 步 warmup → C1 preflight → 四组各 200 updates + val F0/F8 → 汇总
bash scripts/kf/debug_server.sh --preset full --model-kind c0 --devices 2 \
  --warmup-steps 1000 --train-steps 200 --seeds 42
```

每次默认创建独立 `reports/debug-时间戳/`，也可 `--output /your/new/debug-dir` 指定**尚不存在**的目录。主要文件：

- `REPORT.txt`：失败阶段、退出码、最近 20 行输出、排查提示、重试命令。
- `report.json`：解释器、资源路径、各阶段完整 argv/耗时/状态及未执行阶段；不导出完整环境或 HF token。
- `01-environment.log` 等：实时合并 stdout/stderr，完整保留子任务 traceback。
- `environment.json`、`kernels.rank*.json`、`models.json`、`preflight.json`：具体检查结果。
- `retry-阶段.sh`：修复问题后 `bash /实际报告目录/retry-models.sh` 等直接单独重跑；每次使用新目录。它重启该阶段，不恢复 optimizer；训练精确 resume 仍使用前文专用入口。
- `smoke/`、`warmup/`、`matrix/`、`summary/`：所选任务的实际输出。执行过 warmup 后自动将其 C1 传给后续实验，四组使用同一检查点。

例如仅重查数据可执行 `--steps preflight`；仅跑已配置 C1 的矩阵可执行 `--steps rl_preflight,matrix,summary --model-kind c0`。仅重做汇总用 `--steps summary --matrix-root /实际完成的/matrix`。默认每阶段无超时；短检查可加 `--timeout 300`，超时退出码 124 并终止整个 torchrun 进程组，不要给长训练设置过短超时。Ctrl-C 会终止当前子进程组并记录中断。

`models` 阶段核对真实文件的张量结构 hash（与上游 loader registry 一致）；`c0` 模式另检查 C0 内必需 state dict。**这不是完整 GPU 权重加载测试**：完整加载、F0 回归与未来 latent 回归在真实 smoke 中执行。基础模型字节校验另可运行 `download_models.py --verify`。`--preset full` 指 C0 之后的 P0 全流程，尚不包含从基础模型自训 C0 的多卡 PPU 适配。没有 C0 时 preflight 会明确失败，不会跳过并假装训练成功。
