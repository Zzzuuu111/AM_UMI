#!/usr/bin/env python3
"""Extract episodes 60..89 with exact pixels/values and the original holdout."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.sampler import get_val_mask

SOURCE_SHA256 = 'c747a021c573b1a00dbd2c71f3e0147cf269317726bad7cdc331f9df92454d4f'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    manifest_path = Path(str(output) + '.manifest.json')
    partial = Path(str(output) + '.partial')
    if any(p.exists() for p in (output, partial, manifest_path)):
        parser.error('拒绝覆盖数据、未完成的提取文件或 manifest')
    if sha256(source) != SOURCE_SHA256:
        parser.error('源文件不是已验证的最终修正 90 条数据')
    register_codecs()
    output.parent.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(Path(str(source) + '.manifest.json').read_text())
    with zarr.ZipStore(str(source), mode='r') as src_store:
        src = zarr.open_group(src_store, mode='r')
        ends = np.asarray(src['meta/episode_ends'][:])
        if len(ends) != 90 or len(source_manifest['episodes']) != 90:
            parser.error('源文件或 manifest 并非 90 条示教')
        offset, end = int(ends[59]), int(ends[-1])
        n = end - offset
        selected = source_manifest['episodes'][60:90]
        if n != 16208 or not all(
                Path(row['session']).name.startswith(f'vjaw_demo_{91+i:03d}_')
                for i, row in enumerate(selected)):
            parser.error('最后 30 条并非预期的黑圈示教 091..120')
        with zarr.ZipStore(str(partial), mode='w') as dst_store:
            dst = zarr.group(store=dst_store)
            dst.attrs.update({**dict(src.attrs), 'signed_circle30_subset': {
                'source_sha256': SOURCE_SHA256, 'source_episode_indices': list(range(60, 90))}})
            data = dst.create_group('data')
            for name, arr in src['data'].arrays():
                if arr.shape[0] != end:
                    raise ValueError(f'{name}: inconsistent source length')
                if name == 'camera0_rgb':
                    if not isinstance(arr.compressor, zarr.Blosc):
                        raise ValueError('源图像必须使用已核对的无损 Blosc 编码')
                    copied = data.create_dataset(
                        name, shape=(n, *arr.shape[1:]), dtype=arr.dtype,
                        chunks=arr.chunks, compressor=arr.compressor,
                        filters=arr.filters, fill_value=arr.fill_value,
                        order=arr.order, dimension_separator=arr._dimension_separator)
                    for frame in range(0, n, arr.chunks[0]):
                        stop = min(n, frame + arr.chunks[0])
                        copied[frame:stop] = arr[offset+frame:offset+stop]
                else:
                    copied = data.array(
                        name, arr[offset:end], chunks=arr.chunks,
                        compressor=arr.compressor, filters=arr.filters,
                        fill_value=arr.fill_value, order=arr.order)
                if arr.attrs:
                    copied.attrs.update(dict(arr.attrs))
            meta = dst.create_group('meta')
            meta.array('episode_ends', ends[60:] - offset)
        # Check every numerical value and every decoded image pixel.
        with zarr.ZipStore(str(partial), mode='r') as check_store:
            check = zarr.open_group(check_store, mode='r')
            for name, arr in src['data'].arrays():
                copied = check['data'][name]
                if name == 'camera0_rgb':
                    for frame in range(0, n, 64):
                        stop = min(n, frame + 64)
                        np.testing.assert_array_equal(arr[offset+frame:offset+stop], copied[frame:stop])
                else:
                    np.testing.assert_array_equal(arr[offset:end], copied[:])
            np.testing.assert_array_equal(ends[60:] - offset, check['meta/episode_ends'][:])
    os.replace(partial, output)
    manifest = dict(
        schema='am_umi_exact_signed_circle30_subset_v1', offline_only=True,
        source_dataset=source.name, source_sha256=SOURCE_SHA256,
        source_episode_indices=list(range(60, 90)), source_frame_offset=offset,
        episodes=30, frames=n, all_numeric_arrays_equal=True,
        all_decoded_image_frames_equal=True, output_sha256=sha256(output),
        source_split_seed=45, source_val_ratio=0.05,
        val_episode_indices=np.flatnonzero(get_val_mask(90, 0.05, 45)[60:]).tolist(),
        sessions=selected,
        note='Exact values/pixels from the final per-demo-zeroed candidate; lossless Blosc recompression, no new labels.')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'sessions'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
