"""Utilities for EMEET UVC payload-header timing.

Linux exposes ``UVCH`` metadata as a stream of ``uvc_meta_buf`` records:
the host's monotonic reception time, USB SOF, and the original UVC payload
header.  EMEET supplies a PTS on every payload and an SCR source-clock value.
The PTS is constant for all payloads belonging to one video frame, so it can
be used to construct a less jittery camera timeline than Python decode-arrival
time alone.

This is still a *source-to-USB* timing estimate, not a hardware exposure
timestamp.  Both raw host timestamps and the derived timeline are retained.
"""

from __future__ import annotations

import csv
import json
import pathlib

import numpy as np


UVC_STREAM_PTS = 1 << 2
UVC_STREAM_SCR = 1 << 3


def parse_uvch(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Parse concatenated Linux UVCH records without opening any device."""
    data = path.read_bytes()
    offset = invalid = 0
    host_ns: list[int] = []
    sof: list[int] = []
    pts: list[int] = []
    stc: list[int] = []
    scr_sof: list[int] = []
    while offset + 12 <= len(data):
        timestamp_ns = int.from_bytes(data[offset:offset + 8], "little")
        usb_sof = int.from_bytes(data[offset + 8:offset + 10], "little")
        header_len = data[offset + 10]
        if not 2 <= header_len <= 12 or offset + 10 + header_len > len(data):
            invalid += 1
            offset += 1
            continue
        header = data[offset + 10:offset + 10 + header_len]
        flags = header[1]
        cursor = 2
        this_pts = None
        this_stc = None
        this_scr_sof = None
        if flags & UVC_STREAM_PTS:
            if cursor + 4 <= len(header):
                this_pts = int.from_bytes(header[cursor:cursor + 4], "little")
            cursor += 4
        if flags & UVC_STREAM_SCR:
            if cursor + 6 <= len(header):
                this_stc = int.from_bytes(header[cursor:cursor + 4], "little")
                this_scr_sof = int.from_bytes(header[cursor + 4:cursor + 6], "little")
            cursor += 6
        # For frame association PTS is required.  EMEET has both PTS and SCR,
        # but retaining only complete PTS records avoids inventing a frame ID.
        if this_pts is not None:
            host_ns.append(timestamp_ns)
            sof.append(usb_sof)
            pts.append(this_pts)
            stc.append(-1 if this_stc is None else this_stc)
            scr_sof.append(-1 if this_scr_sof is None else this_scr_sof)
        offset += 10 + header_len
    return {
        "host_ns": np.asarray(host_ns, dtype=np.int64),
        "sof": np.asarray(sof, dtype=np.int64),
        "pts": np.asarray(pts, dtype=np.uint64),
        "stc": np.asarray(stc, dtype=np.int64),
        "scr_sof": np.asarray(scr_sof, dtype=np.int64),
        "resync_bytes": np.asarray([invalid], dtype=np.int64),
    }


def unwrap_u32(values: np.ndarray) -> np.ndarray:
    """Unwrap an increasing UVC 32-bit clock into signed 64-bit ticks."""
    if len(values) == 0:
        return np.empty(0, dtype=np.int64)
    result = np.empty(len(values), dtype=np.int64)
    offset = 0
    previous = int(values[0])
    for index, value_raw in enumerate(values):
        value = int(value_raw)
        # Real UVC PTS increments frame to frame.  Treat only a large backwards
        # step as wrap, rather than a malformed isolated header as a reset.
        if index and value < previous and previous - value > (1 << 31):
            offset += 1 << 32
        result[index] = value + offset
        previous = value
    return result


def uvc_frame_events(parsed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reduce payload records to one source-clock event per distinct PTS."""
    pts = parsed["pts"]
    if len(pts) == 0:
        raise RuntimeError("UVCH 中没有带 PTS 的 UVC payload header")
    # PTS is stable for the payloads of one UVC video frame.  Use the first
    # arrival of each PTS, which is closest to when source data entered USB.
    starts = np.r_[True, pts[1:] != pts[:-1]]
    index = np.flatnonzero(starts)
    unwrapped = unwrap_u32(pts[index])
    host = parsed["host_ns"][index].astype(np.float64)
    x = (unwrapped - unwrapped[0]).astype(np.float64)
    if len(x) < 3 or np.ptp(x) <= 0:
        raise RuntimeError("UVCH PTS 帧数不足，无法拟合相机源时钟")
    slope_ns_per_tick, intercept_ns = np.polyfit(x, host, 1)
    mapped_ns = np.rint(slope_ns_per_tick * x + intercept_ns).astype(np.int64)
    residual_ms = (host - mapped_ns) / 1e6
    return {
        "payload_index": index.astype(np.int64),
        "host_ns": parsed["host_ns"][index],
        "pts": pts[index],
        "stc": parsed["stc"][index],
        "mapped_ns": mapped_ns,
        "slope_ns_per_tick": np.asarray([slope_ns_per_tick]),
        "residual_ms": residual_ms,
    }


def _nearest_indices(sorted_values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    right = np.searchsorted(sorted_values, targets, side="left")
    right = np.clip(right, 0, len(sorted_values) - 1)
    left = np.clip(right - 1, 0, len(sorted_values) - 1)
    choose_left = np.abs(targets - sorted_values[left]) <= np.abs(sorted_values[right] - targets)
    return np.where(choose_left, left, right)


def _strict_monotonic_event_indices(
        event_times_ns: np.ndarray, frame_receive_ns: np.ndarray) -> np.ndarray:
    """Minimum-cost ordered one-to-one match to UVC source events.

    A per-frame nearest-neighbour lookup can assign the same PTS event to two
    adjacent decoded frames when their host receive timestamps jitter.  That
    creates a zero-duration camera interval; VIO then has no IMU samples to
    integrate for that frame.  The recorder normally has at least as many UVC
    PTS events as encoded video frames, so enforce a one-to-one ordered match
    and reserve enough remaining events for every remaining video frame.
    """
    n_events = len(event_times_ns)
    n_frames = len(frame_receive_ns)
    if n_events < n_frames:
        raise RuntimeError(
            "UVC PTS 事件数少于视频帧数，无法构造严格递增的相机时间轴："
            f"events={n_events}, frames={n_frames}")
    if n_frames == 0:
        return np.empty(0, dtype=np.int64)
    if np.any(np.diff(event_times_ns) <= 0) or np.any(np.diff(frame_receive_ns) <= 0):
        raise RuntimeError("UVC 事件和视频接收时间都必须严格递增")
    # A greedy nearest match permanently skips events when receive jitter
    # briefly crosses a frame boundary. Subsequent matches can then lead the
    # video by several frames. DP chooses the *whole* ordered assignment.
    # j = frame_index + skipped; skips cannot decrease between frames.
    skips = np.arange(n_events - n_frames + 1, dtype=np.int64)
    if n_frames * len(skips) > 20_000_000:
        raise RuntimeError("UVC 候选事件过多；应先限定有效录制时间窗口")
    parents = np.empty((n_frames, len(skips)), dtype=np.int32)
    costs = ((event_times_ns[skips] - frame_receive_ns[0]).astype(float) / 1e6) ** 2
    for i in range(1, n_frames):
        best = np.minimum.accumulate(costs)
        previous = np.maximum.accumulate(np.where(costs == best, skips, 0))
        parents[i] = previous
        costs = best + ((event_times_ns[i + skips] - frame_receive_ns[i]).astype(float) / 1e6) ** 2
    matched = np.empty(n_frames, dtype=np.int64)
    s = int(np.argmin(costs))
    for i in range(n_frames - 1, -1, -1):
        matched[i] = i + s
        if i:
            s = int(parents[i, s])
    return matched


def preferred_frame_timestamps_path(session_dir: pathlib.Path) -> pathlib.Path:
    """Choose the derived UVC timeline when a session has one."""
    session_dir = pathlib.Path(session_dir)
    derived = session_dir / "frame_timestamps_uvc_source.csv"
    return derived if derived.is_file() else session_dir / "frame_timestamps.csv"


def load_preferred_frame_times(session_dir: pathlib.Path) -> tuple[np.ndarray, pathlib.Path]:
    """Load source-clock relative seconds if available, otherwise raw host time."""
    path = preferred_frame_timestamps_path(session_dir)
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise RuntimeError(f"{path.name} 没有帧")
    key = "uvc_source_relative_s" if "uvc_source_relative_s" in rows[0] else "relative_s"
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64), path


def align_session_frame_timestamps(session_dir: pathlib.Path) -> dict:
    """Write a derived UVC-source timeline for an existing raw session.

    ``frame_timestamps.csv`` remains untouched.  The new CSV associates a
    decoded video frames with distinct UVC source PTS events using a global
    ordered minimum-cost assignment on the shared monotonic clock. This is
    not proof of exposure correspondence. Large residuals remain reported.
    """
    session_dir = session_dir.resolve()
    uvch_path = session_dir / "raw_uvc_payload_headers.bin"
    frame_path = session_dir / "frame_timestamps.csv"
    if not uvch_path.is_file():
        raise FileNotFoundError(f"缺少 UVC 原始元数据：{uvch_path}")
    if not frame_path.is_file():
        raise FileNotFoundError(f"缺少视频帧时间轴：{frame_path}")
    parsed = parse_uvch(uvch_path)
    events = uvc_frame_events(parsed)
    frames: list[dict[str, str]] = []
    with frame_path.open(newline="", encoding="utf-8") as source:
        frames = list(csv.DictReader(source))
    if not frames:
        raise RuntimeError("frame_timestamps.csv 没有帧")
    receive_ns = np.asarray([int(row["receive_monotonic_ns"]) for row in frames], dtype=np.int64)
    # Metadata capture intentionally outlives video writing for a short tail.
    # Restrict candidates to the decoded-video receive window before doing the
    # one-to-one match; otherwise reserving a late tail event forces all frame
    # associations to drift forward over a long recording.
    window_margin_ns = int(50e6)
    in_video_window = ((events["mapped_ns"] >= receive_ns[0] - window_margin_ns) &
                       (events["mapped_ns"] <= receive_ns[-1] + window_margin_ns))
    candidate_event_indices = np.flatnonzero(in_video_window)
    if len(candidate_event_indices) < len(frames):
        # Preserve a usable timeline for unusual devices that deliver fewer
        # metadata events, while making the limitation visible in the report.
        candidate_event_indices = np.arange(len(events["mapped_ns"]), dtype=np.int64)
    candidate_indices = _strict_monotonic_event_indices(
        events["mapped_ns"][candidate_event_indices], receive_ns)
    event_index = candidate_event_indices[candidate_indices]
    source_ns = events["mapped_ns"][event_index]
    source_relative_s = (source_ns.astype(np.float64) - float(source_ns[0])) / 1e9
    delta_ms = (source_ns.astype(np.float64) - receive_ns) / 1e6
    output = session_dir / "frame_timestamps_uvc_source.csv"
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow([
            "frame_index", "receive_monotonic_ns", "relative_s",
            "uvc_pts", "uvc_scr_stc", "uvc_payload_host_monotonic_ns",
            "uvc_source_mapped_monotonic_ns", "uvc_source_relative_s",
            "source_minus_receive_ms",
        ])
        for frame_number, (row, index, source_time, delta) in enumerate(
                zip(frames, event_index, source_ns, delta_ms)):
            writer.writerow([
                row["frame_index"], row["receive_monotonic_ns"], row["relative_s"],
                int(events["pts"][index]), int(events["stc"][index]),
                int(events["host_ns"][index]), int(source_time),
                f"{source_relative_s[frame_number]:.9f}", f"{delta:.6f}",
            ])
    report = {
        "kind": "emeet_uvc_payload_timing_v1",
        "raw_uvch": uvch_path.name,
        "frame_timestamps_input": frame_path.name,
        "frame_timestamps_output": output.name,
        "payload_records_with_pts": int(len(parsed["pts"])),
        "source_frame_events": int(len(events["pts"])),
        "source_events_in_video_window": int(np.count_nonzero(in_video_window)),
        "video_frames": int(len(frames)),
        "pts_clock_ns_per_tick": float(events["slope_ns_per_tick"][0]),
        "pts_clock_hz": float(1e9 / events["slope_ns_per_tick"][0]),
        "source_to_payload_fit_rmse_ms": float(np.sqrt(np.mean(events["residual_ms"] ** 2))),
        "source_minus_decode_receive_median_ms": float(np.median(delta_ms)),
        "source_minus_decode_receive_p95_abs_ms": float(np.percentile(np.abs(delta_ms), 95)),
        "associations_over_50ms": int(np.count_nonzero(np.abs(delta_ms) > 50.0)),
        "frame_event_matching": "strict_monotonic_one_to_one_global_min_squared_cost_v2",
        "strictly_increasing_source_timeline": bool(np.all(np.diff(source_ns) > 0)),
        "note": (
            "UVC SCR/PTS is source-to-USB timing, not a hardware exposure timestamp. "
            "Keep frame_timestamps.csv as the raw host-receive record."
        ),
    }
    (session_dir / "uvc_payload_timing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
