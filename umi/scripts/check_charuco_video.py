"""
Quick charuco visibility check for a calibration video.

Reports how many frames have a usable charuco detection (>= min_corners),
so you can confirm the recording is good BEFORE running the full calibration.

Usage (umi env):
    python scripts/check_charuco_video.py -i deploy_cam_calib.mp4
    python scripts/check_charuco_video.py -i deploy_cam_calib.mp4 --stride 10 --min_corners 14

Good target: >= 60% of checked frames usable, spread across the whole video.
"""
import sys
import click
import cv2
import numpy as np
import av

ROOT_DIR = __file__.rsplit('/', 2)[0]
sys.path.insert(0, ROOT_DIR)

from umi.common.cv_util import get_charuco_board  # noqa: E402


@click.command()
@click.option('-i', '--input', required=True, help='Calibration video')
@click.option('--stride', type=int, default=10, help='Check every N frames')
@click.option('--min_corners', type=int, default=14, help='Min charuco corners to count as usable')
@click.option('--print_bad_every', type=int, default=30, help='Print a marker for every N bad frames')
def main(input, stride, min_corners, print_bad_every):
    board = get_charuco_board()
    detector = cv2.aruco.CharucoDetector(board)

    checked = 0
    usable = 0
    per_marker_hist = {}
    with av.open(input) as container:
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if idx % stride != 0:
                continue
            img = frame.to_ndarray(format='bgr24')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, cids, _, mids = detector.detectBoard(gray)
            n_corners = 0 if cids is None else len(cids)
            n_markers = 0 if mids is None else len(mids)
            checked += 1
            if n_corners >= min_corners:
                usable += 1
            else:
                per_marker_hist[n_markers] = per_marker_hist.get(n_markers, 0) + 1
                if checked % print_bad_every == 0:
                    print(f"frame {idx:6d}: markers={n_markers:2d} corners={n_corners:2d} (bad)")

    pct = 100.0 * usable / max(checked, 1)
    print(f"\nchecked={checked} frames, usable(>={min_corners} corners)={usable} ({pct:.1f}%)")
    print(f"bad-frame marker-count histogram: {dict(sorted(per_marker_hist.items()))}")
    if pct >= 60:
        print("RESULT: OK -> run calibrate_fisheye_intrinsics.py")
    else:
        print("RESULT: BAD -> re-record: board closer (25-40cm), larger in frame, "
              "move slowly, pause at each pose")


if __name__ == "__main__":
    main()
