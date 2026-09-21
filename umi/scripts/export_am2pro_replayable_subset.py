#!/usr/bin/env python3
"""Create a training Zarr containing only accepted arm-replay episodes.

Input is never changed.  The selected episode IDs come from
``filter_am2pro_replayability.py`` and are copied into a new directory-store
Zarr with rebuilt ``meta/episode_ends``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs


def open_source(path: Path):
    if path.is_dir():
        return zarr.open_group(str(path), mode="r"), None
    store = zarr.ZipStore(str(path), mode="r")
    return zarr.open_group(store=store, mode="r"), store


def selected_episode_ids(args):
    if args.episodes is not None:
        return [int(item) for item in args.episodes.split(",") if item.strip()]
    report = json.loads(Path(args.filter_report).expanduser().read_text(encoding="utf-8"))
    if report.get("schema") != "am_umi_am2pro_arm_replay_filter_v1":
        raise ValueError("不是 AM2Pro arm-replay 筛选报告")
    return [int(item) for item in report.get("accepted_episodes", [])]


def main():
    parser = argparse.ArgumentParser(
        description="将 arm-replay 通过的 episode 导出为独立训练 Zarr；不修改源数据。")
    parser.add_argument("--source", required=True, help="源 UMI Zarr 目录或 ZipStore")
    parser.add_argument("--filter-report", default=None,
                        help="filter_am2pro_replayability.py 输出的 JSON 报告")
    parser.add_argument("--episodes", default=None,
                        help="仅用于诊断的逗号分隔 episode ID；与 --filter-report 二选一")
    parser.add_argument("--output", required=True, help="新的目录式 Zarr；必须不存在")
    args = parser.parse_args()
    if (args.filter_report is None) == (args.episodes is None):
        parser.error("必须且只能提供 --filter-report 或 --episodes")

    source_path = Path(args.source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_path}")
    register_codecs()
    root, store = open_source(source_path)
    try:
        if "data" not in root or "meta" not in root or "episode_ends" not in root["meta"]:
            raise ValueError("源 Zarr 必须包含 data、meta 和 meta/episode_ends")
        ends = root["meta"]["episode_ends"][:]
        starts = [0, *map(int, ends[:-1])]
        ids = selected_episode_ids(args)
        if not ids:
            raise ValueError("没有可导出的 accepted episode")
        if len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= len(ends):
            raise ValueError(f"episode ID 无效；合法范围是 0 到 {len(ends) - 1}")

        lengths = [int(ends[index]) - starts[index] for index in ids]
        total = sum(lengths)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out = zarr.open_group(str(output_path), mode="w")
        out.attrs.update(dict(root.attrs))
        out.attrs["am_umi_subset"] = {
            "source": str(source_path), "episode_ids": ids,
            "selection": "AM2Pro offline arm-replay filter",
        }
        src_data = root["data"]
        dst_data = out.create_group("data")
        for name, source in src_data.arrays():
            if source.shape[0] != int(ends[-1]):
                raise ValueError(f"data/{name} 首维与 episode_ends 不一致")
            destination = dst_data.create_dataset(
                name, shape=(total, *source.shape[1:]), dtype=source.dtype,
                chunks=source.chunks, compressor=source.compressor,
                fill_value=source.fill_value, order=source.order)
            offset = 0
            for episode_id, length in zip(ids, lengths):
                start, end = starts[episode_id], int(ends[episode_id])
                destination[offset:offset + length] = source[start:end]
                offset += length
        dst_meta = out.create_group("meta")
        dst_meta.attrs.update(dict(root["meta"].attrs))
        new_ends = []
        cursor = 0
        for length in lengths:
            cursor += length
            new_ends.append(cursor)
        dst_meta.array("episode_ends", new_ends, dtype=ends.dtype)
        (output_path / "am2pro_subset_manifest.json").write_text(
            json.dumps({"source": str(source_path), "accepted_episode_ids": ids,
                        "episode_lengths": lengths}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
    finally:
        if store is not None:
            store.close()
    print("AM2PRO_REPLAYABLE_SUBSET_OK")
    print("source:", source_path)
    print("episodes:", ids)
    print("frames:", total)
    print("output:", output_path)


if __name__ == "__main__":
    main()
