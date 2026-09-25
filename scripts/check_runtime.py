"""Check pinned runtime provenance without creating a simulator or CUDA context."""

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import platform
from pathlib import Path

from uav_gap.runtime import ROOT, assert_environment

PACKAGES = [
    'torch', 'numpy', 'torchvision', 'pytorch3d', 'warp-lang', 'isaacgym',
    'aerial-gym', 'gym', 'gymnasium', 'rl-games', 'sample-factory',
    'matplotlib', 'ninja', 'uav-visual-gap',
]
MODULES = ['isaacgym', 'aerial_gym', 'uav_gap']


def build_report():
    assert_environment()
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError('Required distribution is missing: ' + name) from exc
    import torch

    origins = {}
    for name in MODULES:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError('Required Python module cannot be located: ' + name)
        origin = Path(spec.origin).resolve()
        try:
            origins[name] = origin.relative_to(ROOT).as_posix()
        except ValueError as exc:
            raise RuntimeError('Module is not loaded from this project: ' + name) from exc
    return {
        'python_version': platform.python_version(),
        'platform': platform.system() + '-' + platform.machine(),
        'torch_cuda_build': torch.version.cuda,
        'packages': packages,
        'module_origins_project_relative': origins,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-inventory', action='store_true')
    args = parser.parse_args()
    report = build_report()
    encoded = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.write_inventory:
        output = (ROOT / 'runs/runtime_inventory.json').resolve()
        output.relative_to(ROOT)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding='utf-8')
    print(encoded, end='')


if __name__ == '__main__':
    main()
