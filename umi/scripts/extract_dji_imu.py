"""
Extract DJI (Osmo Action 4) gyro + accelerometer from an MP4 into imu_data.json.

DJI stores IMU in two private Protobuf timed-metadata streams inside the MP4:
  - `dbgi` (codec_tag "dbgi")  : gyroscope (angular velocity)
  - `djmd` (codec_tag "djmd")  : metadata (accelerometer etc.)

This script reverse-engineers the Protobuf structure (verified against a
DJI_2026..._0005_D.MP4 from an Osmo Action 4 / AC203):

  dbgi stream
    field 1   : header (proto name "dbginfo_ac203.proto", sensor OV48C40)
    field 2   : repeated packet (one per video frame)
      field 1 : packet body
        field 1  : frame timestamp (varint, ~6 bytes)
        field 4  : gyro block
          field 200 : gyro meta (19 bytes)
          field 201 : gyro samples (840 bytes, little-endian int16 interleaved
                      with an incrementing sequence counter)
        field 5  : accel block
          field 2  : accel meta (289 bytes)
          field 3  : accel samples (3275 bytes)

NOTE / TODO:
  * The exact scale factor (raw int16 -> rad/s for gyro, m/s^2 for accel) and
    the exact sample layout still need to be calibrated against Gyroflow's
    output for this exact model. This script extracts the RAW samples and the
    per-frame timestamps; the caller can then apply the scale.

Usage:
    python scripts/extract_dji_imu.py -i <video.mp4> -o <imu_data.json>
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

# %%
import pathlib
import json
import subprocess
import tempfile
import click
import numpy as np


# %%
def _read_varint(buf, i):
    r = 0
    s = 0
    while True:
        b = buf[i]
        i += 1
        r |= (b & 0x7f) << s
        if not (b & 0x80):
            break
        s += 7
    return r, i


def _parse_fields(buf):
    """Parse a protobuf message into [(field_num, wire_type, value)]."""
    out = []
    i = 0
    while i < len(buf):
        tag, i = _read_varint(buf, i)
        fn = tag >> 3
        wt = tag & 7
        if wt == 0:
            v, i = _read_varint(buf, i)
            out.append((fn, wt, v))
        elif wt == 1:
            out.append((fn, wt, buf[i:i + 8]))
            i += 8
        elif wt == 2:
            ln, i = _read_varint(buf, i)
            out.append((fn, wt, buf[i:i + ln]))
            i += ln
        elif wt == 5:
            out.append((fn, wt, buf[i:i + 4]))
            i += 4
        else:
            break  # unknown wire type
    return out


def _get_fields(msg, field_num):
    return [v for fn, wt, v in msg if fn == field_num and wt == 2]


def _extract_stream(video_path, stream_index, out_path):
    """Extract a data stream via ffmpeg."""
    cmd = [
        'ffmpeg', '-v', 'error', '-y',
        '-i', str(video_path),
        '-map', f'0:{stream_index}',
        '-c', 'copy', '-f', 'data', str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _find_streams(video_path):
    """Return {codec_tag: stream_index} for the data streams."""
    cmd = ['ffprobe', '-v', 'error', '-show_streams', str(video_path)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    streams = []
    cur = {}
    for line in out.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            cur[k] = v
        elif line == '[STREAM]':
            cur = {}
        elif line == '[/STREAM]':
            streams.append(cur)
    result = {}
    for i, s in enumerate(streams):
        if s.get('codec_type') == 'data':
            result[s.get('codec_tag_string', '')] = i
    return result


@click.command()
@click.option('-i', '--input', required=True, help='DJI MP4 video')
@click.option('-o', '--output', default=None, help='Output imu_data.json')
def main(input, output):
    input = pathlib.Path(os.path.expanduser(input))
    assert input.is_file(), f"Input not found: {input}"
    if output is None:
        output = input.with_suffix('.imu_data.json')
    output = pathlib.Path(os.path.expanduser(output))

    streams = _find_streams(input)
    print(f"Data streams: {streams}")
    assert 'dbgi' in streams, "No dbgi (gyro) stream found"
    dbgi_idx = streams['dbgi']
    djmd_idx = streams.get('djmd')

    with tempfile.TemporaryDirectory() as tmp:
        dbgi_path = pathlib.Path(tmp) / 'dbgi.bin'
        _extract_stream(input, dbgi_idx, dbgi_path)
        dbgi = dbgi_path.read_bytes()

        # parse top-level: field 1 = header, field 2 = repeated packet
        top = _parse_fields(dbgi)
        packets = _get_fields(top, 2)
        print(f"Parsed {len(packets)} gyro packets from dbgi")

        gyro_samples = []   # list of (timestamp, raw_int16_array)
        accel_samples = []
        for pkt in packets:
            body = _get_fields(_parse_fields(pkt), 1)
            if not body:
                continue
            body = _parse_fields(body[0])
            # timestamp
            ts_fields = _get_fields(body, 1)
            ts = ts_fields[0] if ts_fields else b''
            # gyro: field 4 -> field 201
            gyro_blocks = _get_fields(body, 4)
            gyro_raw = b''
            if gyro_blocks:
                g201 = _get_fields(_parse_fields(gyro_blocks[0]), 201)
                if g201:
                    gyro_raw = g201[0]
            # accel: field 5 -> field 3
            accel_blocks = _get_fields(body, 5)
            accel_raw = b''
            if accel_blocks:
                a3 = _get_fields(_parse_fields(accel_blocks[0]), 3)
                if a3:
                    accel_raw = a3[0]

            # int16 conversion: truncate to an even length (some packets are off-by-one)
            gyro_raw = gyro_raw[: (len(gyro_raw) // 2) * 2]
            gyro_samples.append({
                'timestamp_bytes': ts.hex(),
                'raw_int16': np.frombuffer(gyro_raw, dtype='<i2').tolist() if gyro_raw else [],
            })
            if accel_raw:
                accel_samples.append({
                    'timestamp_bytes': ts.hex(),
                    'raw_bytes': accel_raw.hex(),
                })

        accel_lens = sorted({len(a['raw_bytes']) for a in accel_samples})
        result = {
            'camera': 'DJI_AC203',
            'streams': streams,
            'n_gyro_packets': len(gyro_samples),
            'gyro_raw_len_per_packet': len(gyro_samples[0]['raw_int16']) if gyro_samples else 0,
            'gyro_sample_first': gyro_samples[0] if gyro_samples else None,
            'accel_raw_len_per_packet': accel_lens,
            'note': ("RAW int16 gyro samples; scale factor (raw->rad/s) and accel "
                     "layout still need calibration against Gyroflow for AC203."),
        }

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved -> {output}")
        print(f"  gyro raw int16 per packet: {result['gyro_raw_len_per_packet']}")
        print(f"  accel raw byte lengths: {accel_lens}")
        print(f"  first packet timestamp bytes: {gyro_samples[0]['timestamp_bytes'] if gyro_samples else 'N/A'}")


# %%
if __name__ == "__main__":
    main()
