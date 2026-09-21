"""Generate a derived camera-source timeline from a saved EMEET UVCH stream."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from imu_work.uvc_payload_timing import align_session_frame_timestamps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="由已保存的 EMEET UVC PTS/SCR 元数据重建视频帧时间轴")
    parser.add_argument("--session-dir", required=True)
    args = parser.parse_args()
    session = pathlib.Path(args.session_dir).expanduser().resolve()
    report = align_session_frame_timestamps(session)
    print("UVC_PAYLOAD_TIME_ALIGNMENT_OK")
    print("session:", session)
    print("source_frame_events:", report["source_frame_events"],
          "video_frames:", report["video_frames"])
    print("PTS_clock_hz:", f"{report['pts_clock_hz']:.1f}")
    print("p95_source_vs_decode_ms:",
          f"{report['source_minus_decode_receive_p95_abs_ms']:.3f}")
    print("output:", session / report["frame_timestamps_output"])


if __name__ == "__main__":
    main()
