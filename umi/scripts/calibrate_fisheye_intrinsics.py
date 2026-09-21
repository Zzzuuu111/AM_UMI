"""
Calibrate a fisheye camera (e.g. DJI Osmo Action 4) from a Charuco board video,
and output UMI-format intrinsics JSON + validation images.

The ``-o`` path is the active intrinsics file. Each run first writes a
timestamped candidate under ``<output parent>/candidates/``. After evaluation,
the active file is replaced automatically only if the candidate has a strictly
smaller reprojection error; the old active file is archived first.

Usage:
    python scripts/calibrate_fisheye_intrinsics.py \
        -i <calibration_video.mp4> \
        -o calibration/osmo4_intrinsics_2_7k.json

The Charuco board geometry must match the printed board (see
scripts/gen_charuco_board.py / umi.common.cv_util.get_charuco_board).
Defaults: 8x5 grid, 30mm squares, 18mm tags, DICT_4X4_100 with id offset 50.

Memory note: this script keeps only ONE full-resolution frame in memory
(the single best frame for validation), so it will not blow up RAM on long
2.7K/4K videos.
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import json
import datetime
import shutil
import click
import numpy as np
import cv2
import av
from tqdm import tqdm
import scipy.optimize
import scipy.sparse

from umi.common.cv_util import (
    get_charuco_board,
    FisheyeRectConverter,
)

# %%
def _match_image_points(board, charuco_corners, charuco_ids):
    """Return (obj_pts (N,1,3), img_pts (N,1,2)) in meters.

    charuco_ids from CharucoDetector.detectBoard are indices into
    board.getChessboardCorners(), so we index those directly. Do NOT use
    board.matchImagePoints(): on OpenCV 4.7 it returns marker corners with
    broken correspondence (28 charuco corners -> 80 points), which feeds
    garbage object points into cv2.fisheye.calibrate and crashes it with
    'fabs(norm_u1) > 0'.
    """
    chess_corners = np.asarray(board.getChessboardCorners(), dtype=np.float32)  # (N,3)
    obj_pts = chess_corners[charuco_ids.flatten()].reshape(-1, 1, 3)
    img_pts = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
    return obj_pts, img_pts


def _refine_corners(gray, corners):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return refined


def _mean_reproj_err(obj_list, img_list, rvecs, tvecs, K, D):
    """Per-image mean reprojection error (px)."""
    intr = (K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0, 0], D[1, 0], D[2, 0], D[3, 0])
    errs = []
    for obj, img, rvec, tvec in zip(obj_list, img_list, rvecs, tvecs):
        uv = _fisheye_project(
            obj.reshape(-1, 3).astype(np.float64), rvec, tvec, intr)
        errs.append(np.linalg.norm(uv - img.reshape(-1, 2).astype(np.float64), axis=1).mean())
    return np.array(errs)


def _is_degenerate_board(img_pts, min_spread_px=10.0):
    """True if detected corners are nearly collinear (board seen edge-on).

    Such frames crash cv2.fisheye.calibrate's InitExtrinsics with
    'fabs(norm_u1) > 0', so we drop them before calibration.
    """
    pts = img_pts[:, 0, :].astype(np.float64)
    if len(pts) < 4:
        return True
    centered = pts - pts.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    # s[1] is the spread perpendicular to the best-fit line
    return float(s[1]) < min_spread_px


def _fisheye_project(points3d, rvec, tvec, intr):
    """Project (N,3) board-frame points to (N,2) pixels with the Kannala-Brandt
    fisheye model (identical to cv2.fisheye.projectPoints).

    intr = (fx, fy, cx, cy, k1, k2, k3, k4).
    """
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    k1, k2, k3, k4 = intr[4], intr[5], intr[6], intr[7]
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    Xc = points3d @ R.T + np.asarray(tvec, dtype=np.float64).reshape(3)
    x = Xc[:, 0] / Xc[:, 2]
    y = Xc[:, 1] / Xc[:, 2]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)
    theta2 = theta * theta
    theta_d = theta * (1 + k1 * theta2 + k2 * theta2 ** 2 + k3 * theta2 ** 3 + k4 * theta2 ** 4)
    scale = np.where(r > 1e-10, theta_d / r, 1.0)
    u = fx * scale * x + cx
    v = fy * scale * y + cy
    return np.stack([u, v], axis=1)


def _fisheye_bundle_adjust(obj_list, img_list, image_size, K_init, D_init,
                           rvecs_init, tvecs_init, robust=True, multi_start=False):
    """Bundle-adjust fisheye intrinsics + per-frame extrinsics with scipy.

    Uses a SINGLE focal length f (fx = fy): camera pixels are square, so fy must
    equal fx; leaving them free lets fy collapse to a degenerate small value.

    Returns (K (3,3), D (4,1), rvecs (list (3,1)), tvecs (list (3,1))).
    This replaces cv2.fisheye.calibrate, whose InitExtrinsics homography is
    fragile for strong-fisheye / large-board frames (OpenCV 4.7 bug).
    """
    nf = len(obj_list)
    objs = [o.reshape(-1, 3).astype(np.float64) for o in obj_list]
    imgs = [im.reshape(-1, 2).astype(np.float64) for im in img_list]
    n_resid = sum(2 * len(o) for o in objs)
    # params: f, cx, cy, k1..k4, then (rvec, tvec) per frame
    n_params = 7 + 6 * nf
    w0, h0 = image_size

    def build_p0(f):
        p0 = np.zeros(n_params)
        p0[0] = f
        p0[1] = K_init[0, 2]; p0[2] = K_init[1, 2]
        p0[3:7] = D_init.reshape(4)
        for i in range(nf):
            p0[7 + 6 * i: 7 + 6 * i + 3] = np.asarray(rvecs_init[i]).reshape(3)
            p0[7 + 6 * i + 3: 7 + 6 * i + 6] = np.asarray(tvecs_init[i]).reshape(3)
        return p0

    offsets = np.zeros(nf + 1, dtype=np.int64)
    for i in range(nf):
        offsets[i + 1] = offsets[i] + 2 * len(objs[i])

    def resid(p):
        res = np.empty(n_resid)
        intr = (p[0], p[0], p[1], p[2], p[3], p[4], p[5], p[6])
        for i in range(nf):
            rv = p[7 + 6 * i: 7 + 6 * i + 3]
            tv = p[7 + 6 * i + 3: 7 + 6 * i + 6]
            uv = _fisheye_project(objs[i], rv, tv, intr)
            res[offsets[i]:offsets[i + 1]] = (uv - imgs[i]).reshape(-1)
        return res

    # sparse Jacobian pattern: each point depends on the 7 intrinsics + its own
    # frame's 6 extrinsics only (block structure -> fast finite differences)
    rows, cols = [], []
    for i in range(nf):
        npts = len(objs[i])
        base = offsets[i]
        col_intr = np.arange(7)
        col_extr = 7 + 6 * i + np.arange(6)
        for r in range(base, base + 2 * npts):
            rows.extend([r] * 7); cols.extend(col_intr.tolist())
            rows.extend([r] * 6); cols.extend(col_extr.tolist())
    sparsity = scipy.sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_resid, n_params))

    # bounds: k1 may be ~0 (equidistant) or slightly positive (stereographic);
    # keep k2..k4 small to avoid overfitting.
    lower = np.full(n_params, -np.inf)
    upper = np.full(n_params, np.inf)
    lower[0] = 300.0; upper[0] = 2500.0        # f
    lower[1] = 0.3 * w0; upper[1] = 0.7 * w0   # cx
    lower[2] = 0.3 * h0; upper[2] = 0.7 * h0   # cy
    lower[3] = -0.5; upper[3] = 0.5            # k1
    lower[4:7] = -1.0; upper[4:7] = 1.0        # k2, k3, k4

    f0 = K_init[0, 0]
    scales = [0.6, 0.8, 1.0, 1.2, 1.4] if multi_start else [1.0]
    best = None
    for s in scales:
        p0 = build_p0(f0 * s)
        sol = scipy.optimize.least_squares(
            resid, p0, jac_sparsity=sparsity, x_scale='jac',
            bounds=(lower, upper),
            loss='soft_l1' if robust else 'linear',
            method='trf', max_nfev=200)
        if best is None or sol.cost < best.cost:
            best = sol
    sol = best

    p = sol.x
    f = p[0]
    K = np.array([[f, 0, p[1]], [0, f, p[2]], [0, 0, 1]], dtype=np.float64)
    D = p[3:7].reshape(4, 1)
    rvecs = [p[7 + 6 * i: 7 + 6 * i + 3].reshape(3, 1) for i in range(nf)]
    tvecs = [p[7 + 6 * i + 3: 7 + 6 * i + 6].reshape(3, 1) for i in range(nf)]
    return K, D, rvecs, tvecs, sol


def _fisheye_intrinsics_to_umi(K, D, image_size, fps, n_images, reproj_error):
    """Convert OpenCV fisheye K/D to UMI (openicc) JSON format."""
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return {
        "final_reproj_error": float(reproj_error),
        "fps": float(fps),
        "image_height": int(image_size[1]),
        "image_width": int(image_size[0]),
        "intrinsic_type": "FISHEYE",
        "intrinsics": {
            "aspect_ratio": fy / fx,
            "focal_length": fx,
            "principal_pt_x": cx,
            "principal_pt_y": cy,
            "radial_distortion_1": float(D[0, 0]),
            "radial_distortion_2": float(D[1, 0]),
            "radial_distortion_3": float(D[2, 0]),
            "radial_distortion_4": float(D[3, 0]),
            "skew": float(K[0, 1]),
        },
        "nr_calib_images": int(n_images),
        "stabelized": False,
    }


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Calibration video (Charuco board)')
@click.option(
    '-o', '--output', required=True,
    help=('Active UMI intrinsics JSON. Each run first writes a timestamped candidate; '
          'the active file is replaced only if the candidate has lower reprojection error.'))
@click.option('-to', '--tag_id_offset', type=int, default=50)
@click.option('-sq', '--square_length_mm', type=float, default=30.0)
@click.option('-tg', '--tag_length_mm', type=float, default=18.0)
@click.option('--stride', type=int, default=5, help='Detect board every N frames')
@click.option('--max_frames', type=int, default=200, help='Max frames used (evenly subsampled)')
@click.option('--min_corners', type=int, default=14, help='Min Charuco corners to accept a frame')
@click.option('--rect_fov', type=float, default=110.0, help='Output FOV (deg) for rectified validation image')
def main(input, output, tag_id_offset, square_length_mm, tag_length_mm,
         stride, max_frames, min_corners, rect_fov):
    # limit OpenCV threads to avoid pegging the machine during detection
    cv2.setNumThreads(min(4, os.cpu_count() or 1))

    input = pathlib.Path(os.path.expanduser(input))
    active_output = pathlib.Path(os.path.expanduser(output))
    assert input.is_file(), f"Input video not found: {input}"
    active_output.parent.mkdir(parents=True, exist_ok=True)

    # Never overwrite the active calibration while evaluating a new one.  Keep the
    # full candidate (JSON + visual validation images) under a timestamped folder,
    # then promote it only when its final reprojection error is strictly lower.
    active_error = None
    if active_output.is_file():
        try:
            with open(active_output, 'r') as f:
                active_error = float(json.load(f)['final_reproj_error'])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"Warning: could not read active calibration metric from {active_output}: {e}")
            print("Treating this run as the first valid calibration; the existing file will be backed up.")

    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    candidate_dir = active_output.parent / 'candidates' / run_id
    candidate_dir.mkdir(parents=True, exist_ok=False)
    output = candidate_dir / active_output.name
    if active_error is None:
        print(f"Active calibration: none/read-error; candidate will be evaluated at {candidate_dir}")
    else:
        print(f"Active calibration: {active_output} ({active_error:.4f} px)")
        print(f"Candidate calibration: {candidate_dir}")

    board = get_charuco_board(
        tag_id_offset=tag_id_offset,
        grid_size=(8, 5),
        square_length_mm=square_length_mm,
        tag_length_mm=tag_length_mm)
    detector = cv2.aruco.CharucoDetector(board)

    # ---------- stage 1: detect corners over the whole video ----------
    print(f"Detecting Charuco board (stride={stride}, min_corners={min_corners}) ...")
    all_obj_pts = []
    all_img_pts = []
    frame_idxs = []
    n_degenerate = 0

    # keep a bounded set (3) of full-res frames for validation: the calibration
    # itself still uses ALL detected frames; these images are only for eyeballing.
    cand_best = {'key': -1, 'img': None, 'obj': None, 'imgpts': None, 'frame': -1}      # most corners
    cand_sparse = {'key': 10**9, 'img': None, 'obj': None, 'imgpts': None, 'frame': -1}  # fewest corners
    cand_edge = {'key': -1.0, 'img': None, 'obj': None, 'imgpts': None, 'frame': -1}    # board farthest from center

    def _set_cand(cand, key, img, obj_pts, img_pts, i):
        cand['key'] = key
        cand['img'] = img
        cand['obj'] = obj_pts
        cand['imgpts'] = img_pts
        cand['frame'] = i

    with av.open(str(input)) as container:
        in_stream = container.streams.video[0]
        in_stream.thread_type = "AUTO"
        fps = float(in_stream.average_rate)
        image_size = (in_stream.width, in_stream.height)
        img_center = np.array([in_stream.width / 2, in_stream.height / 2])

        for i, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames):
            if i % stride != 0:
                continue
            img = frame.to_ndarray(format='rgb24')
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
            if charuco_ids is None or len(charuco_ids) < min_corners:
                continue

            refined = _refine_corners(gray, charuco_corners)
            obj_pts, img_pts = _match_image_points(board, refined, charuco_ids)

            # drop frames whose corners are nearly collinear (board seen at a
            # grazing angle) — they crash cv2.fisheye InitExtrinsics
            if _is_degenerate_board(img_pts):
                n_degenerate += 1
                continue

            all_obj_pts.append(obj_pts)
            all_img_pts.append(img_pts)
            frame_idxs.append(i)

            n = len(charuco_ids)
            if n > cand_best['key']:
                _set_cand(cand_best, n, img, obj_pts, img_pts, i)
            if n < cand_sparse['key']:
                _set_cand(cand_sparse, n, img, obj_pts, img_pts, i)
            board_center = img_pts[:, 0, :].mean(axis=0)
            d = float(np.linalg.norm(board_center - img_center))
            if d > cand_edge['key']:
                _set_cand(cand_edge, d, img, obj_pts, img_pts, i)

    n_detected = len(all_obj_pts)
    if n_degenerate:
        print(f"Skipped {n_degenerate} degenerate frame(s) (board seen nearly edge-on).")
    print(f"Detected board in {n_detected} frames.")
    if n_detected < 8:
        raise RuntimeError(
            f"Only {n_detected} usable frames found (< 8). "
            "Check board is flat, well-lit, and fills the frame (esp. edges).")

    # ---------- stage 2: evenly subsample to max_frames ----------
    if n_detected > max_frames:
        keep = np.linspace(0, n_detected - 1, max_frames).round().astype(int)
        keep = np.unique(keep)
        all_obj_pts = [all_obj_pts[k] for k in keep]
        all_img_pts = [all_img_pts[k] for k in keep]
        frame_idxs = [frame_idxs[k] for k in keep]
        print(f"Subsampled to {len(all_obj_pts)} frames.")

    # force-include the validation candidate frames (dedup by frame idx) so their
    # rvec/tvec exist after calibration
    candidates = [cand_best, cand_sparse, cand_edge]
    labels = ['best', 'sparse', 'edge']
    seen_frames = set(frame_idxs)
    for cand, label in zip(candidates, labels):
        if cand['frame'] < 0:
            continue
        if cand['frame'] in seen_frames:
            continue
        all_obj_pts.append(cand['obj'])
        all_img_pts.append(cand['imgpts'])
        frame_idxs.append(cand['frame'])
        seen_frames.add(cand['frame'])

    # ---------- stage 3: calibrate (scipy bundle adjustment + outlier rejection) ----------
    print("Running pinhole init + scipy fisheye bundle adjustment ...")
    # Step 0: pinhole calibration gives per-frame extrinsics. Its fx is unreliable
    # for a fisheye, so we use a direct equidistant estimate for the focal length.
    try:
        _, K_pin, _, rvecs_pin, tvecs_pin = cv2.calibrateCamera(
            all_obj_pts, all_img_pts, image_size, None, None)
    except cv2.error as e:
        raise RuntimeError(f"pinhole init calibration failed: {e}") from e
    print(f"  pinhole init fx={K_pin[0, 0]:.1f} fy={K_pin[1, 1]:.1f}")

    obj_list = list(all_obj_pts)
    img_list = list(all_img_pts)
    fidx_list = list(frame_idxs)
    # equidistant fisheye focal-length estimate: r = f * theta, theta_max = FOV/2.
    # assume ~155 deg diagonal FOV; multi-start covers the uncertainty.
    w0, h0 = image_size
    f_fish = 0.5 * np.hypot(w0, h0) / np.deg2rad(155.0 / 2.0)
    K_cur = np.array([[f_fish, 0, K_pin[0, 2]],
                      [0, f_fish, K_pin[1, 2]],
                      [0, 0, 1]], dtype=np.float64)
    D_cur = np.zeros((4, 1), dtype=np.float64)
    rv_cur = list(rvecs_pin)
    tv_cur = list(tvecs_pin)

    K = D = None
    rvecs = tvecs = None
    per_img_err = np.array([])
    for it in range(3):
        # bundle adjust (robust loss + multi-start on first pass; plain after)
        K, D, rvecs, tvecs, sol = _fisheye_bundle_adjust(
            obj_list, img_list, image_size, K_cur, D_cur, rv_cur, tv_cur,
            robust=(it == 0), multi_start=(it == 0))
        per_img_err = _mean_reproj_err(obj_list, img_list, rvecs, tvecs, K, D)
        if len(per_img_err) <= 8:
            break
        # drop gross outlier frames (bad corners / degenerate views)
        thr = max(1.0, 3.0 * float(np.median(per_img_err)))
        keep = per_img_err <= thr
        n_drop = int((~keep).sum())
        if n_drop == 0:
            break
        print(f"  iter {it + 1}: dropped {n_drop} outlier frame(s) (err > {thr:.2f} px)")
        obj_list = [obj_list[i] for i in range(len(obj_list)) if keep[i]]
        img_list = [img_list[i] for i in range(len(img_list)) if keep[i]]
        fidx_list = [fidx_list[i] for i in range(len(fidx_list)) if keep[i]]
        # seed the next pass with the current (filtered) refined result
        K_cur = K
        D_cur = D
        rv_cur = [rvecs[i] for i in range(len(rvecs)) if keep[i]]
        tv_cur = [tvecs[i] for i in range(len(tvecs)) if keep[i]]

    # adopt the final (filtered) set
    all_obj_pts = obj_list
    all_img_pts = img_list
    frame_idxs = fidx_list
    overall_err = float(per_img_err.mean())

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    w, h = image_size

    print("\n========== Calibration result ==========")
    print(f"overall reproj error : {overall_err:.4f} px")
    print(f"per-image err range  : {per_img_err.min():.4f} ~ {per_img_err.max():.4f} px")
    print(f"fx / fy              : {fx:.2f} / {fy:.2f}  (ratio {fy / fx:.5f})")
    print(f"cx / cy              : {cx:.2f} / {cy:.2f}  (center {w / 2:.1f} / {h / 2:.1f})")
    print(f"k1..k4               : {D[0,0]:.6f}, {D[1,0]:.6f}, {D[2,0]:.6f}, {D[3,0]:.6f}")
    print("=========================================")
    print("Sanity hints:")
    if abs(fx - fy) / fx > 0.02:
        print("  [!] fx/fy differ > 2% (unusual for fisheye) -> check board flatness/coverage")
    if abs(cx - w / 2) / w > 0.05 or abs(cy - h / 2) / h > 0.05:
        print("  [!] principal point far from image center -> check for biased corner coverage")
    if overall_err < 0.5:
        print("  [OK] reproj error < 0.5 px (GoPro reference ~0.29 px)")
    elif overall_err < 1.0:
        print("  [~]  reproj error 0.5~1.0 px, acceptable but inspect undistortion edges")
    else:
        print("  [!!] reproj error > 1.0 px -> re-record, especially edge coverage")

    # ---------- stage 5: write JSON ----------
    umi_json = _fisheye_intrinsics_to_umi(K, D, image_size, fps, len(all_obj_pts), overall_err)
    with open(output, 'w') as f:
        json.dump(umi_json, f, indent=2)
    print(f"\nSaved candidate intrinsics -> {output}")

    # ---------- stage 6: validation images ----------
    out_dir = output.parent

    # resolve each candidate's position in the calibrated arrays
    resolved = []
    for cand, label in zip(candidates, labels):
        if cand['frame'] < 0:
            continue
        if cand['frame'] not in frame_idxs:
            continue  # dropped as an outlier during calibration
        k = frame_idxs.index(cand['frame'])
        resolved.append({
            'label': label,
            'img': cand['img'],
            'obj': all_obj_pts[k],
            'imgpts': all_img_pts[k],
            'rvec': rvecs[k],
            'tvec': tvecs[k],
            'err': per_img_err[k],
        })

    # (a) per-candidate: undistorted comparison (full-FOV) + corner overlay
    for r in resolved:
        orig = r['img']
        # undistorted comparison
        undist_full = cv2.fisheye.undistortImage(orig, K, D, Knew=K)
        comparison = np.concatenate([orig, undist_full], axis=1)
        cmp_path = out_dir.joinpath(f'calib_undist_comparison_{r["label"]}.png')
        cv2.imwrite(str(cmp_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

        # corner overlay: detected (blue) vs projected (red)
        vis_img = orig.copy()
        intr = (K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0, 0], D[1, 0], D[2, 0], D[3, 0])
        proj_pts = _fisheye_project(
            r['obj'].reshape(-1, 3).astype(np.float64), r['rvec'], r['tvec'], intr)
        for pt in r['imgpts'][:, 0, :]:
            cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 4, (255, 0, 0), -1)  # blue detected
        for pt in proj_pts:
            cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), 1)  # red projected
        cv2.drawFrameAxes(vis_img, K, D, r['rvec'], r['tvec'], 0.05)
        overlay_path = out_dir.joinpath(f'calib_corner_overlay_{r["label"]}.png')
        cv2.imwrite(str(overlay_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        print(f"[{r['label']}] reproj err={r['err']:.3f} px -> "
              f"{cmp_path.name}, {overlay_path.name}")

    # (b) pinhole-rectified image (straight-line check) on the best frame
    best_r = resolved[0]
    intr_dict = {'K': K.astype(np.float64), 'D': D.astype(np.float64),
                 'DIM': np.array([w, h], dtype=np.int64)}
    converter = FisheyeRectConverter(**intr_dict, out_size=(w, h), out_fov=rect_fov)
    rect = converter.forward(best_r['img'])
    rect_path = out_dir.joinpath('calib_rectified.png')
    cv2.imwrite(str(rect_path), cv2.cvtColor(rect, cv2.COLOR_RGB2BGR))
    print(f"Saved pinhole-rectified image (fov={rect_fov}) -> {rect_path}")

    # (d) per-image reprojection error plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(per_img_err)), per_img_err, color='steelblue')
        ax.axhline(overall_err, color='red', linestyle='--',
                   label=f'mean = {overall_err:.3f} px')
        ax.set_xlabel('frame index (subsampled)')
        ax.set_ylabel('mean reprojection error (px)')
        ax.set_title('Per-image reprojection error')
        ax.legend()
        fig.tight_layout()
        err_path = out_dir.joinpath('calib_reproj_error.png')
        fig.savefig(str(err_path), dpi=120)
        plt.close(fig)
        print(f"Saved reprojection error plot -> {err_path}")
    except ImportError:
        print("matplotlib not available, skipped error plot.")

    # ---------- stage 7: automatically promote only an objectively better result ----------
    accepted = active_error is None or overall_err < active_error
    summary = {
        'run_id': run_id,
        'candidate_intrinsics': str(output),
        'active_intrinsics': str(active_output),
        'candidate_final_reproj_error': overall_err,
        'active_final_reproj_error_before': active_error,
        'comparison': 'candidate < active (strict)',
        'accepted_as_active': accepted,
    }
    summary_path = candidate_dir / 'candidate_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    if accepted:
        if active_output.is_file():
            history_dir = active_output.parent / 'history'
            history_dir.mkdir(parents=True, exist_ok=True)
            backup_path = history_dir / f'{active_output.stem}_{run_id}{active_output.suffix}'
            shutil.copy2(active_output, backup_path)
            print(f"Backed up previous active intrinsics -> {backup_path}")
        shutil.copy2(output, active_output)
        if active_error is None:
            print(f"ACCEPTED: first valid candidate is now active -> {active_output}")
        else:
            print(f"ACCEPTED: {overall_err:.4f} px < {active_error:.4f} px; active file updated -> {active_output}")
    else:
        print(f"REJECTED: {overall_err:.4f} px >= {active_error:.4f} px; active file unchanged -> {active_output}")
    print(f"Candidate record kept -> {candidate_dir}")


# %%
if __name__ == "__main__":
    main()
