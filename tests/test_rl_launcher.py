"""Validate launch arguments and secret handling without starting Ray or GPUs."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize('smoke', ['0', '1'])
def test_launcher_runtime_and_arguments(tmp_path, smoke):
    root = Path(__file__).resolve().parents[1]
    project = tmp_path / 'project'
    scripts = project / 'scripts'
    scripts.mkdir(parents=True)
    launcher = scripts / 'run_reasoning_state_rl.sh'
    shutil.copyfile(root / 'scripts' / launcher.name, launcher)
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    ray = binaries / 'ray'
    ray.write_text('''#!/usr/bin/env python3
import json, os, pathlib, stat, sys
if sys.argv[1] != 'job':
    raise SystemExit(0)
arg = next(a for a in sys.argv if a.startswith('--runtime-env='))
p = pathlib.Path(arg.split('=', 1)[1])
assert stat.S_IMODE(p.stat().st_mode) == 0o600
values = json.loads(p.read_text())['env_vars']
assert values['ANSWER_API_KEY'] == os.environ['ANSWER_API_KEY']
args = sys.argv
assert '--custom-rm-path' in args
assert args[args.index('--custom-rm-path')+1] == 'evomemstate.reasoning_state_reward.reward_from_reasoning_state'
json.loads(args[args.index('--apply-chat-template-kwargs')+1])
pathlib.Path(os.environ['LAUNCH_REPORT']).write_text(json.dumps({'runtime_file':str(p)}))
print('mock training submission validated')
''')
    ray.chmod(0o755)
    nvidia = binaries / 'nvidia-smi'
    nvidia.write_text('#!/bin/sh\nexit 0\n')
    nvidia.chmod(0o755)
    for name in ('hf', 'ref', 'megatron'):
        (tmp_path / name).mkdir()
    data = tmp_path / 'external.jsonl'
    data.write_text('{}\n')
    secret = 'test-only-secret"\\`$()'
    report = tmp_path / 'report.json'
    env = dict(os.environ, PATH=str(binaries)+os.pathsep+os.environ['PATH'],
               SLIME_DIR=str(root/'slime'), HF_CHECKPOINT=str(tmp_path/'hf'),
               REF_TORCH_DIST=str(tmp_path/'ref'), MEGATRON_PATH=str(tmp_path/'megatron'),
               PROMPT_DATA=str(data), SMOKE=smoke, ANSWER_API_KEY=secret,
               LAUNCH_REPORT=str(report), DATA_SAVED_PATH=str(tmp_path/'output'))
    result = subprocess.run(['bash', str(launcher)], env=env, text=True,
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'mock training submission validated' in result.stdout
    assert secret not in result.stdout + result.stderr
    runtime = Path(json.loads(report.read_text())['runtime_file'])
    assert not runtime.exists()
    for log in (tmp_path/'output').rglob('*.log'):
        assert secret not in log.read_text()
