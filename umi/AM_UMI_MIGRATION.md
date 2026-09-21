# AM_UMI: LeRobot + UMI

This directory is a source-only UMI copy colocated with the `AM_UMI` copy of
`lerobot_alohamini`. It is intended to run in the cloned Conda environment of
the same name, which keeps the LeRobot package versions authoritative.

## Layout

- `../` — cloned `lerobot_alohamini` repository.
- `./` — copied UMI source, configurations, scripts, UMI package, and the
  `ORB_SLAM3_umi` and `spnav` source trees.
- `./imu_work/` — only IMU Python tools. Capture files, calibration outputs,
  datasets, videos, W&B logs, and other generated artefacts remain in the
  original `universal_manipulation_interface` repository.

## Use

```bash
conda activate AM_UMI
cd /home/zzzjh/AM_UMI
pip install -e .
cd umi
pip install -e spnav
python -c "import torch, lerobot, umi, diffusion_policy; print(torch.__version__)"
```

`requirements-am-umi.txt` lists UMI-only additions. It must not be replaced by
the old `environment_umi_full.yml`: that file is a Python 3.9 / Torch 2.1
environment and would downgrade the LeRobot stack.

## Compatibility policy

AM_UMI runs Python 3.12 and preserves the installed LeRobot dependency stack,
including its Torch, TorchVision, NumPy, Diffusers, Accelerate, TIMM, Zarr,
Numcodecs, OpenCV, and W&B versions. UMI's original pins are therefore not
applied where they conflict. The UMI source has no hard-coded Python 3.9 or
Torch 2.1 pin; runtime compatibility is verified with imports and the UMI
training-config smoke test.

Hardware still requires the host-side services and devices used by UMI (for
example `spacenavd` / `libspnav`, UVC cameras, and the robot controller). They
are system dependencies, not Python environment packages.

## Included test data

`test_data/replay_case_0099/` contains a copied, read-only AM2Pro replay
dataset (`replay_case_0099_dataset.zarr`, 18 MB) and its dataset plan. It has
three episodes (829 frames), 224x224 RGB observations, end-effector poses, and
gripper widths. It is suitable for AM2Pro replay and for testing the UMI
dataset loader; it is not a general-purpose training corpus.

For a low-memory or debugging run of the UMI normalizer, set
`UMI_NORMALIZER_WORKERS=0`. The default remains a bounded four-worker scan.

The bundled `spnav` source has an import shim so UMI can be run directly from
this repository without its outer source directory masking the installed
SpaceMouse client package.

## 8 GB GPU profile

The included `experiment/am_umi_gpu_8gb.yaml` profile uses batch size one,
FP16, gradient accumulation of eight, and disables the EMA model copy. Use it
with the copied test dataset as follows:

```bash
conda activate AM_UMI
cd /home/zzzjh/AM_UMI/umi
export UMI_NORMALIZER_WORKERS=0
python train.py --config-name train_diffusion_unet_image_workspace \
  task=umi task.dataset_path=test_data/replay_case_0099/replay_case_0099_dataset.zarr \
  +experiment=am_umi_gpu_8gb
```

## Conservative 4 GB GPU test profile

For a validation run that must leave substantial GPU headroom, use
`experiment/am_umi_gpu_4gb.yaml` instead.  It uses the same FP16, batch-one
setup, and additionally caps this Python process at 45% of the GPU allocator
(about 3.4 GiB on the available 8 GB card).  It deliberately uses stateless
SGD rather than AdamW: AdamW's two full-size optimizer-state tensors do not
fit alongside this policy and its gradients below 4 GB.  This profile validates
the full data, forward, backward, and parameter-update path, but is not a
replacement for an AdamW training run.

```bash
conda activate AM_UMI
cd /home/zzzjh/AM_UMI/umi
export UMI_NORMALIZER_WORKERS=0
python scripts/test_umi_gpu_4gb.py
```

The test script does not start W&B, real cameras/robot code, rollouts, or
checkpoint writing. It performs exactly one data-load, FP16 forward, backward,
and optimizer-update step using the bundled replay data.

## Archived-checkpoint smoke tests

The copied scripts first verify a strict CPU restore, then one FP32 GPU
inference using the bundled Zarr replay. Both skip robot, camera, W&B, and
checkpoint writes. The GPU inference has the same conservative 45% GPU
allocator cap as the 4 GB validation profile:

```bash
cd /home/zzzjh/AM_UMI/umi
python scripts/test_umi_checkpoint_restore.py
python scripts/test_umi_checkpoint_gpu_inference.py
```
