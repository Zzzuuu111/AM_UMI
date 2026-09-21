#!/usr/bin/env python3
"""Merge screened AM_UMI sessions into one multi-episode ZipStore dataset.

The policy training dataset loader accepts one ``.zarr.zip`` ReplayBuffer,
whereas the V-jaw recorder emits one directory ``processed.zarr`` per
session.  This tool preserves every episode boundary and validates the schema
before it writes anything.  It is offline-only: it never imports or connects
to robot-control code.
"""

import argparse
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.replay_buffer import ReplayBuffer


REQUIRED_KEYS = (
    "action",
    "camera0_rgb",
    "robot0_eef_pos",
    "robot0_eef_rot_axis_angle",
    "robot0_gripper_width",
    "robot0_demo_start_pose",
    "robot0_demo_end_pose",
)


@contextmanager
def open_root(path: Path):
    """Yield a Zarr root and close the backing ZipStore when applicable."""
    store = None
    if path.is_dir():
        root = zarr.open_group(str(path), mode="r")
    else:
        store = zarr.ZipStore(str(path), mode="r")
        root = zarr.group(store=store)
    try:
        yield root
    finally:
        if store is not None:
            store.close()


def resolve_sources(list_path: Path) -> list[Path]:
    sources = []
    for line_number, raw_line in enumerate(list_path.read_text(encoding="utf-8").splitlines(), 1):
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(
                f"{list_path}:{line_number} 指向的 zarr 不存在: {candidate}"
            )
        sources.append(candidate)
    if not sources:
        raise ValueError(f"{list_path} 中没有可合并的候选 zarr")
    if len(set(sources)) != len(sources):
        raise ValueError("候选列表含重复 zarr；请先去重，避免训练样本被重复计数")
    return sources


def inspect_source(path: Path, expected_schema: dict | None) -> tuple[dict, list[int]]:
    """Check one source without materializing its image frames."""
    with open_root(path) as root:
        if "data" not in root or "meta" not in root or "episode_ends" not in root["meta"]:
            raise ValueError(f"{path}: 必须包含 data、meta 和 meta/episode_ends")
        data = root["data"]
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(f"{path}: 缺少必需数组: {missing}")
        keys = tuple(sorted(data.array_keys()))
        ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        if len(ends) == 0 or np.any(ends <= 0) or np.any(np.diff(ends) <= 0):
            raise ValueError(f"{path}: meta/episode_ends 必须是严格递增的正整数")
        total_frames = int(ends[-1])
        schema = {}
        for key in keys:
            array = data[key]
            if array.ndim < 1 or int(array.shape[0]) != total_frames:
                raise ValueError(f"{path}: data/{key} 的帧数与 episode_ends 不一致")
            schema[key] = {
                "shape_tail": tuple(array.shape[1:]),
                "dtype": str(array.dtype),
            }
        if expected_schema is not None and schema != expected_schema:
            raise ValueError(
                f"{path}: 数组名称、尾部形状或 dtype 与第一个候选不一致；拒绝合并"
            )
        return schema, ends.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 accepted_sessions.txt 中的离线合格 zarr 合并为训练用 .zarr.zip（不连接机器人）"
    )
    parser.add_argument("--input-list", required=True, help="每行一个 processed.zarr 的文本列表")
    parser.add_argument("--output", required=True, help="新的训练集路径，必须以 .zarr.zip 结尾")
    args = parser.parse_args()

    list_path = Path(args.input_list).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not list_path.is_file():
        parser.error(f"--input-list 不存在: {list_path}")
    if not output.name.endswith(".zarr.zip"):
        parser.error("--output 必须以 .zarr.zip 结尾，训练器只读取 ZipStore")
    if output.exists():
        parser.error(f"拒绝覆盖已有训练集: {output}")
    partial_dir = output.with_name(output.name + ".partial.dir")
    partial_zip = output.with_name(output.name + ".partial")
    if partial_dir.exists() or partial_zip.exists():
        parser.error(
            "发现上次未完成的临时输出: "
            f"{partial_dir if partial_dir.exists() else partial_zip}；请人工检查后再处理"
        )

    register_codecs()
    sources = resolve_sources(list_path)
    schema = None
    inspected = []
    for source in sources:
        schema, episode_ends = inspect_source(source, schema)
        inspected.append({"source": str(source), "episode_ends": episode_ends})

    output.parent.mkdir(parents=True, exist_ok=True)
    # ZipStore cannot rename/rewrite entries, while ReplayBuffer.add_episode
    # resizes episode_ends.  Build in a temporary DirectoryStore first, then
    # make one immutable zip copy after all episodes are present.
    output_store = zarr.DirectoryStore(str(partial_dir))
    try:
        buffer = ReplayBuffer.create_empty_zarr(storage=output_store)
        for source in sources:
            with open_root(source) as root:
                data = root["data"]
                ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
                starts = np.r_[0, ends[:-1]]
                chunks = {key: data[key].chunks for key in data.array_keys()}
                compressors = {key: data[key].compressor for key in data.array_keys()}
                for start, end in zip(starts, ends):
                    episode = {key: np.asarray(data[key][int(start):int(end)])
                               for key in data.array_keys()}
                    buffer.add_episode(episode, chunks=chunks, compressors=compressors)
        merged_episodes = int(buffer.n_episodes)
        merged_frames = int(buffer.n_steps)
        output_store.close()
        output_store = None
        zip_store = zarr.ZipStore(str(partial_zip), mode="w")
        try:
            zarr.copy_store(
                source=zarr.DirectoryStore(str(partial_dir)),
                dest=zip_store,
                if_exists="replace",
            )
        finally:
            zip_store.close()
        os.replace(partial_zip, output)
        shutil.rmtree(partial_dir)
    finally:
        if output_store is not None:
            output_store.close()

    manifest = {
        "schema": "am_umi_training_merge_v1",
        "offline_only": True,
        "input_list": str(list_path),
        "output": str(output),
        "source_sessions": inspected,
        "episodes": merged_episodes,
        "frames": merged_frames,
        "arrays": {key: {"shape_tail": list(value["shape_tail"]), "dtype": value["dtype"]}
                   for key, value in schema.items()},
    }
    manifest_path = output.with_name(output.name + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("AM_UMI_TRAINING_MERGE_OK")
    print(f"sources: {len(sources)}; episodes: {merged_episodes}; frames: {merged_frames}")
    print("dataset:", output)
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
