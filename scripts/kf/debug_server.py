"""Stage runner: one command, separate logs, fail-fast reports; stdlib only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import selectors
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PRESETS = {'check': ['environment', 'kernels', 'models'],
           'smoke': ['environment', 'kernels', 'models', 'preflight', 'smoke'],
           'full': ['environment', 'kernels', 'models', 'preflight', 'smoke', 'warmup', 'rl_preflight', 'matrix', 'summary']}
HINTS = {
    'environment': '检查当前 Python/厂商 torch 路径、依赖版本和 traceback 中第一个导入错误；PPU 用 install_ppu.py，勿安装根 requirements.txt。',
    'kernels': '单卡失败查看 SDPA/RoPE/Conv3D/RNG；仅多卡失败检查设备可见性与厂商通信库。此阶段不加载模型。',
    'download': '检查网络/HF_ENDPOINT、磁盘空间及仓库权限；重复 download 会沿用版本锁。',
    'models': '检查 DIFFSYNTH_MODEL_BASE_PATH、文件分片及结构 hash；base 模式要基础模型，c0 模式要完整驾驶 C0。',
    'preflight': '查看 preflight.json 的 errors：C0、数据、地图、manifest、文本/metric cache 是否齐全匹配。基础模型不能代替 C0。',
    'smoke': '查看日志中最后一个子命令，以及 smoke 内部各 run 的 status.json；区分 OOM、模型加载、梯度、评分错误。',
    'warmup': '查看 warmup/status.json 和 train_metrics.jsonl；检查 C0、冻结参数、显存和梯度。',
    'rl_preflight': '检查共同 C1 是否来自当前 C0，以及 split/statistics hash 是否一致。',
    'matrix': '查看 matrix 下具体 k/f/seed 的 status.json；成功子实验不等于四组均成功。',
    'summary': '确认 matrix 的八个 val 评测均完成；summary 不填补缺失结果。',
}
ENV_KEYS = ['SIMWAM_RUNTIME', 'DIFFSYNTH_MODEL_BASE_PATH', 'SIMWAM_IL_CHECKPOINT', 'SIMWAM_KF_CHECKPOINT',
            'NAVSIM_LOG_PATH', 'NAVSIM_SENSOR_BLOBS_PATH', 'NAVSIM_TEST_LOG_PATH', 'NAVSIM_TEST_SENSOR_BLOBS_PATH',
            'NAVSIM_METRIC_CACHE_PATH', 'NAVSIM_VAL_METRIC_CACHE_PATH', 'NAVSIM_TEST_METRIC_CACHE_PATH',
            'NAVSIM_TEXT_EMBED_CACHE', 'NAVSIM_STATS_PATH', 'NAVSIM_DEVKIT_ROOT', 'NUPLAN_MAPS_ROOT',
            'NUPLAN_MAP_VERSION', 'KF_TRAIN_MANIFEST', 'KF_VAL_MANIFEST', 'KF_TEST_MANIFEST', 'NPROC_PER_NODE']


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def commands(a, out):
    py = sys.executable
    return {
        'environment': [py, 'scripts/kf/check_server.py', '--output', str(out / 'environment.json')],
        'kernels': [py, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(a.devices),
                    'scripts/kf/check_server.py', '--kernels', '--output', str(out / 'kernels.json')],
        'download': [py, 'scripts/kf/download_models.py', '--root', os.environ.get('DIFFSYNTH_MODEL_BASE_PATH', '/path/to/models')],
        'models': [py, 'scripts/kf/check_models.py', '--kind', a.model_kind, '--output', str(out / 'models.json')],
        'preflight': [py, 'scripts/kf/preflight.py', '--stage', 'warmup', '--output', str(out / 'preflight.json')],
        'smoke': [py, 'scripts/kf/smoke.py', '--output-root', str(out / 'smoke')],
        'warmup': ['bash', 'scripts/kf/warmup.sh', '--max-steps', str(a.warmup_steps), '--run-dir', str(out / 'warmup')],
        'rl_preflight': [py, 'scripts/kf/preflight.py', '--stage', 'rl', '--output', str(out / 'rl_preflight.json')],
        'matrix': [py, 'scripts/kf/matrix.py', '--stage', 'pilot', '--seeds', a.seeds,
                   '--max-steps', str(a.train_steps), '--output-root', str(out / 'matrix')],
        'summary': [py, 'scripts/kf/summarize.py', '--runs-root', str(Path(a.matrix_root).resolve() if a.matrix_root else out / 'matrix'), '--output-dir', str(out / 'summary')],
    }


def run_logged(cmd, log, env, timeout=0):
    """Stream both outputs; kill the entire torchrun process group on interruption."""
    started = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    timed_out = False
    def stop():
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
    try:
        with log.open('wb') as stream, selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            while selector.get_map():
                if timeout and time.monotonic() - started > timeout:
                    timed_out = True; stop()
                for key, _ in selector.select(timeout=0.2):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data: selector.unregister(key.fileobj); continue
                    stream.write(data); stream.flush()
                    sys.stdout.write(data.decode('utf-8', errors='replace')); sys.stdout.flush()
            return (124 if timed_out else proc.wait()), time.monotonic() - started
    except BaseException:
        stop(); raise
    finally:
        proc.stdout.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', choices=PRESETS, default='check')
    p.add_argument('--steps', help='Comma-separated stages; overrides preset. download must be explicitly selected.')
    p.add_argument('--output', help='New diagnostic directory; existing paths are never overwritten')
    p.add_argument('--devices', type=int, choices=[1, 2, 4, 8], default=1)
    p.add_argument('--model-kind', choices=['base', 'c0'], default='base')
    p.add_argument('--warmup-steps', type=int, default=1000)
    p.add_argument('--train-steps', type=int, default=200)
    p.add_argument('--seeds', default='42')
    p.add_argument('--matrix-root', help='Existing completed matrix for a summary-only run')
    p.add_argument('--timeout', type=float, default=0, help='Per-stage seconds; 0 means unlimited')
    p.add_argument('--plan', action='store_true', help='Write command plan; execute nothing, no model/data required')
    a = p.parse_args()
    if a.warmup_steps < 1 or a.train_steps < 1 or a.timeout < 0: p.error('Invalid step count/timeout')
    try:
        if any(int(s) < 0 for s in a.seeds.split(',')): raise ValueError()
    except ValueError: p.error('seeds must be comma-separated nonnegative integers')
    stages = a.steps.split(',') if a.steps else PRESETS[a.preset]
    if 'matrix' in stages and a.train_steps % 4: p.error('matrix train-steps must be a multiple of 4 (complete rollout boundary)')
    if len(set(stages)) != len(stages) or any(s not in HINTS for s in stages): p.error('Unknown/duplicate stages: ' + ','.join(stages))
    out = Path(a.output or f'reports/debug-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}').resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    env['PYTHONUNBUFFERED'] = '1'; env['PYTHONNOUSERSITE'] = '1'; env['NPROC_PER_NODE'] = str(a.devices)
    paths = [str(ROOT / 'src'), str(ROOT / 'navsim')]
    if env.get('SIMWAM_RUNTIME') == 'ppu': paths.insert(0, str(ROOT / 'configs/server/compat'))
    env['PYTHONPATH'] = os.pathsep.join(paths)
    env['NAVSIM_DEVKIT_ROOT'] = str(ROOT / 'navsim')
    table = commands(a, out)
    report = {'status': 'planned' if a.plan else 'running', 'root': str(ROOT), 'python': sys.executable,
              'real_execution': not a.plan, 'environment': {k: env[k] for k in ENV_KEYS if k in env}, 'stages': []}
    def save():
        write_json(out / 'report.json', report)
        lines = [f"Status: {report['status']}", f'Python: {sys.executable}', '']
        for row in report['stages']:
            lines.append(f"{row['stage']}: {row['status']} (exit={row.get('exit_code', '-')})")
            if row['status'] in ('failed', 'interrupted'):
                lines += ['Log: ' + row['log'], 'Next: ' + HINTS[row['stage']], 'Retry: ' + row['retry']]
                lines += ['Recent output:'] + row.get('log_tail', [])
        (out / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
    save()
    try:
        for index, stage in enumerate(stages):
            if stage in ('rl_preflight', 'matrix') and 'warmup' in stages[:index]:
                env['SIMWAM_KF_CHECKPOINT'] = str(out / 'warmup/export/kf_policy.pt')
            cmd = table[stage]; log = out / f'{index+1:02d}-{stage}.log'
            retry = [sys.executable, 'scripts/kf/debug_server.py', '--steps', stage, '--devices', str(a.devices),
                     '--model-kind', a.model_kind, '--warmup-steps', str(a.warmup_steps), '--train-steps', str(a.train_steps), '--seeds', a.seeds]
            if stage == 'summary': retry += ['--matrix-root', str(Path(a.matrix_root).resolve() if a.matrix_root else out / 'matrix')]
            retry_script = out / f'retry-{stage}.sh'
            exports = '\n'.join(f'export {k}={shlex.quote(env[k])}' for k in ENV_KEYS if k in env)
            retry_script.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd ' + shlex.quote(str(ROOT)) + '\n' + exports + '\nexec ' + shlex.join(retry) + '\n')
            row = {'stage': stage, 'status': 'planned' if a.plan else 'running', 'command': cmd,
                   'log': str(log), 'retry': 'bash ' + shlex.quote(str(retry_script))}
            report['stages'].append(row); save()
            print(f'\n[{stage}] {shlex.join(cmd)}\nLog: {log}', flush=True)
            if a.plan: continue
            code, elapsed = run_logged(cmd, log, env, a.timeout)
            row.update(exit_code=code, elapsed_seconds=round(elapsed, 3), status='passed' if code == 0 else 'failed')
            if code and log.exists():
                with log.open('rb') as tail:
                    tail.seek(max(0, log.stat().st_size - 8192))
                    row['log_tail'] = tail.read().decode('utf-8', errors='replace').splitlines()[-20:]
            save()
            if code:
                report['status'] = 'failed'; report['not_executed'] = stages[index+1:]; save()
                print(f'\nFAILED: {stage}\n{HINTS[stage]}\nReport: {out / "REPORT.txt"}\nRetry: {row["retry"]}')
                return 1
        report['status'] = 'planned' if a.plan else 'passed'; save()
        print(f'\nReport: {out / "REPORT.txt"}'); return 0
    except (KeyboardInterrupt, Exception) as exc:
        report['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        if report['stages'] and report['stages'][-1]['status'] == 'running': report['stages'][-1]['status'] = report['status']
        save(); print(f'Error: {exc}; report: {out / "REPORT.txt"}', file=sys.stderr); return 130 if isinstance(exc, KeyboardInterrupt) else 1

if __name__ == '__main__': sys.exit(main())
