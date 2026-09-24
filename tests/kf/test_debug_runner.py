import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]

def runner():
    spec = importlib.util.spec_from_file_location('debug_server', ROOT / 'scripts/kf/debug_server.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def test_full_plan_executes_nothing_and_keeps_secrets_out(tmp_path):
    out = tmp_path / 'plan'
    env = dict(os.environ, HF_TOKEN='DO_NOT_RECORD_ME')
    result = subprocess.run([sys.executable, 'scripts/kf/debug_server.py', '--preset', 'full', '--plan', '--output', str(out)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((out / 'report.json').read_text())
    assert report['status'] == 'planned' and report['real_execution'] is False
    assert len(report['stages']) == 9
    assert not list(out.glob('*.log'))
    assert all('DO_NOT_RECORD_ME' not in f.read_text() for f in out.iterdir())
    assert str(out / 'warmup/export/kf_policy.pt') in (out / 'retry-matrix.sh').read_text()

def test_logged_exit_code_and_stderr(tmp_path):
    log = tmp_path / 'failed.log'
    code, _ = runner().run_logged([sys.executable, '-c', 'import sys; print("diagnostic",file=sys.stderr); sys.exit(7)'], log, dict(os.environ))
    assert code == 7 and 'diagnostic' in log.read_text()

def test_timeout_stops_process(tmp_path):
    code, elapsed = runner().run_logged([sys.executable, '-c', 'import time; time.sleep(30)'], tmp_path / 'timeout.log', dict(os.environ), .2)
    assert code == 124 and elapsed < 15

def test_fail_fast_report_and_retry(tmp_path, monkeypatch):
    r = runner(); out = tmp_path / 'fail'
    monkeypatch.setattr(sys, 'argv', ['debug_server.py', '--steps', 'environment,models', '--output', str(out)])
    def fail(cmd, log, env, timeout):
        log.write_text('simulated dependency failure'); return 3, .1
    monkeypatch.setattr(r, 'run_logged', fail)
    assert r.main() == 1
    report = json.loads((out / 'report.json').read_text())
    assert report['status'] == 'failed'
    assert report['stages'][0]['exit_code'] == 3
    assert report['not_executed'] == ['models']
    assert (out / 'retry-environment.sh').is_file()

def test_existing_output_not_overwritten(tmp_path):
    out = tmp_path / 'existing'; out.mkdir(); (out / 'keep').write_text('keep')
    result = subprocess.run([sys.executable, 'scripts/kf/debug_server.py', '--plan', '--output', str(out)], cwd=ROOT, capture_output=True)
    assert result.returncode != 0 and (out / 'keep').read_text() == 'keep'
