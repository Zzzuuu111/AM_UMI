#!/usr/bin/env python3
"""Download the private epoch-60/circle30 artifacts and verify every checksum."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def matches(path, item):
    if not path.is_file() or path.stat().st_size != item['bytes']:
        return False
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest() == item['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'example/vjaw_circle30_epoch60_finetune_assets.json').read_text())
    for item in manifest['assets']:
        target = (ROOT / item['local_path']).resolve()
        target.relative_to(ROOT)
        if matches(target, item):
            print('VERIFIED', item['local_path'], flush=True)
            continue
        if args.verify_only:
            raise SystemExit(f'文件缺失或校验失败: {target}')
        if target.exists():
            raise SystemExit(f'拒绝覆盖校验不符的现有文件，请先检查或移出: {target}')
        from huggingface_hub import hf_hub_download
        target.parent.mkdir(parents=True, exist_ok=True)
        print('DOWNLOADING', item['repo_id'], item['filename'], flush=True)
        downloaded = hf_hub_download(
            repo_id=item['repo_id'], repo_type=item['repo_type'],
            filename=item['filename'], revision=item.get('revision', 'main'),
            endpoint='https://huggingface.co', local_dir=target.parent)
        if Path(downloaded).resolve() != target or not matches(target, item):
            raise RuntimeError(f'下载文件校验失败: {target}')
        print('VERIFIED', item['local_path'], flush=True)
    print('ASSETS_READY: bash scripts/start_vjaw_circle30_signed_finetune.sh --smoke')


if __name__ == '__main__':
    main()
