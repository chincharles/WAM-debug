# Source from the SimWAM-KF project root. Fill remaining /path/to values.
# Dataset paths migrated from vla/configs/navsim/aliyun_paths.sh; remote existence unverified.
export OPENSCENE_DATA_ROOT=/mnt/cpfs-wlc-rdma-300t/navsim/openscene-v1.1
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/map"
export NAVSIM_DEVKIT_ROOT="$PWD/navsim"
export NAVSIM_LOG_PATH="$OPENSCENE_DATA_ROOT/navsim_logs/trainval"
export NAVSIM_SENSOR_BLOBS_PATH="$OPENSCENE_DATA_ROOT/sensor_blobs/trainval"
export NAVSIM_TEST_LOG_PATH="$OPENSCENE_DATA_ROOT/navsim_logs/test"
export NAVSIM_TEST_SENSOR_BLOBS_PATH="$OPENSCENE_DATA_ROOT/sensor_blobs/test"
export NAVSIM_METRIC_CACHE_PATH=/path/to/metric_cache_train
export NAVSIM_VAL_METRIC_CACHE_PATH=/path/to/metric_cache_val
export NAVSIM_TEST_METRIC_CACHE_PATH=/path/to/metric_cache_test
export NAVSIM_TEXT_EMBED_CACHE=/path/to/text_embedding_cache
export NAVSIM_STATS_PATH=/path/to/navsim_dataset_stats.json
export SIMWAM_IL_CHECKPOINT=/path/to/simwam_il.pt
export SIMWAM_KF_CHECKPOINT="$PWD/runs/kf/warmup/export/kf_policy.pt"
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/pretrained_models
export KF_TRAIN_MANIFEST="$PWD/manifests/train.jsonl"
export KF_VAL_MANIFEST="$PWD/manifests/val.jsonl"
export KF_TEST_MANIFEST="$PWD/manifests/test.jsonl"
export NPROC_PER_NODE=1
export PYTHONPATH="$PWD/src:$PWD/navsim:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Optional mini dataset paths; P0 train/test defaults above remain unchanged.
export NAVSIM_MINI_LOG_PATH="$OPENSCENE_DATA_ROOT/navsim_logs/mini"
export NAVSIM_MINI_SENSOR_BLOBS_PATH="$OPENSCENE_DATA_ROOT/sensor_blobs/mini"
