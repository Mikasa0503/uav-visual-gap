"""Install the pinned platform-free HCSP Iris render asset project-locally."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HCSP_COMMIT = '009961b8f5702dd0c1c943cef0e01e09dfcd138d'
IRIS_SHA256 = 'e96833fe3d768c4bd449b02e473dcfe692e46198eaa9f3b91b10a8963a2d1b77'
COLOR_REFERENCE_SHA256 = '43bd07b25de3dc8c63babf05543fd3db05694fbabfae12f7f0f2001ffe4d0abe'
LICENSE_SHA256 = '8d5bb214ab1010733bb1f6b91435b40c0f9255312bec99233e834cf8c63ddcf1'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path,
                        help='Existing local checkout of https://github.com/thu-uav/HCSP')
    args = parser.parse_args()
    source = args.source.resolve()
    commit = subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], check=True,
        capture_output=True, text=True).stdout.strip()
    if commit != HCSP_COMMIT:
        raise ValueError('Expected HCSP commit %s, got %s' % (HCSP_COMMIT, commit))
    iris = source/'hcsp/robots/assets/usd/iris.usd'
    color_reference = source/'hcsp/robots/assets/usd/iris_batVisualOnly.usd'
    license_path = source/'LICENSE'
    if (sha256(iris) != IRIS_SHA256 or sha256(color_reference) != COLOR_REFERENCE_SHA256
            or sha256(license_path) != LICENSE_SHA256):
        raise ValueError('HCSP Iris, color reference, or license hash does not match the pinned source')
    destination = ROOT/'assets/hcsp_iris'
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(iris, destination/'iris.usd')
    shutil.copy2(license_path, destination/'LICENSE')
    manifest = {
        'source': 'https://github.com/thu-uav/HCSP',
        'commit': HCSP_COMMIT,
        'asset': 'hcsp/robots/assets/usd/iris.usd',
        'asset_sha256': IRIS_SHA256,
        'color_reference': 'hcsp/robots/assets/usd/iris_batVisualOnly.usd',
        'color_reference_sha256': COLOR_REFERENCE_SHA256,
        'color_reference_imported': False,
        'license': 'MIT',
        'license_sha256': LICENSE_SHA256,
        'mesh_objects': 5,
        'striking_platform': False,
        'purpose': 'offline render visual only; simulation collision and dynamics unchanged',
    }
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
