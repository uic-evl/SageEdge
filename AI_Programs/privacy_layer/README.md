# privacy_layer

A privacy-preserving image and video capture plugin for the
[Sage / Waggle](https://sagecontinuum.org) edge-computing environment.
Detects human faces in real time and applies a configurable mask &mdash;
box blur, Gaussian blur, median blur, pixelation, or full block-out
&mdash; on the device, before any frames are saved or transmitted.

## How it works

1. A frame is read from the configured source (Waggle camera, RTSP
   stream, or video file).
2. The frame is optionally downscaled for detection only &mdash; the
   full-resolution frame is preserved for masking.
3. [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
   loaded through OpenCV's DNN module, detects every face in the frame.
4. Each face box is padded outward (default 15%) to cover hairlines,
   ears, and detection wobble.
5. The selected masking method is applied to each padded region.
6. The blurred frame is discarded, written to a local mp4, or uploaded
   as a periodic snapshot to Beehive, depending on output mode.

YuNet is ~300 KB and runs on CPU at hundreds of frames per second.
There is no PyTorch or CUDA dependency.

## Files

| File | Purpose |
|---|---|
| `main.py` | Plugin entry point &mdash; argument parsing, camera open, frame loop |
| `privacy_layer.py` | Face detector loader, detection function, blur method registry |
| `Dockerfile` | Container build &mdash; CPU-only, multi-arch |
| `requirements.txt` | Python dependencies |
| `sage.yaml` | Sage plugin manifest for ECR |
| `ecr-meta/` | Long-form description (Science / AI@Edge / Ontology) |

## Configuration

Every setting is exposed as both a CLI flag (`--name`) and an environment
variable (`NAME`). Environment variables are the typical interface for
plugins running in containers.

### Source

| Variable | Default | Notes |
|---|---|---|
| `STREAM` | _empty_ | File path or `rtsp://...` / `http://...` URL. Takes precedence over `CAMERA`. |
| `CAMERA` | _empty_ | Named Waggle camera (e.g. `top`, `left`). Used when running on a Sage node. |
| `SNAPSHOT_ONLY` | `0` | If `1`, capture one frame and exit. |

### Detection

| Variable | Default | Notes |
|---|---|---|
| `CONF` | `0.6` | Face detection confidence threshold (0&ndash;1). |
| `DETECT_WIDTH` | `640` | Downscale frames to this width before detection. `0` = detect at full resolution. |

### Masking

| Variable | Default | Options / Notes |
|---|---|---|
| `METHOD` | `box` | `box`, `gaussian`, `median`, `pixelate`, `solid`. |
| `BLUR_STRENGTH` | `25` | Higher = more aggressive obscuring. Ignored for `solid`. |
| `PAD_FRAC` | `0.15` | Fraction of padding around each face box. |

### Output

| Variable | Default | Notes |
|---|---|---|
| `OUTPUT` | `none` | `none`, `file`, `upload`, or `both`. |
| `OUT_PATH` | `blurred.mp4` | Output filename when `OUTPUT` includes `file`. Saved under `output/<timestamp>/`. |
| `UPLOAD_EVERY` | `150` | Upload one snapshot every N frames when `OUTPUT` includes `upload`. |
| `PUBLISH_COUNT` | `0` | If `1`, publish per-frame face count to Beehive. |

### Masking method comparison

- `box` &mdash; default. Cheap averaging filter; visually close to Gaussian
  at matching strength. Recommended for most deployments.
- `gaussian` &mdash; smooth and natural-looking. The most expensive method
  per frame.
- `median` &mdash; posterized "oil painting" effect. Preserves face-region
  edges.
- `pixelate` &mdash; classic mosaic censor. Most visually recognizable as
  a privacy effect.
- `solid` &mdash; full black mask. Strongest privacy guarantee.

## Building

```bash
docker build -t privacy-layer .
```

The build pre-downloads the YuNet ONNX model into the image so the
container runs offline on edge nodes.

## Running

### On a video file

```bash
docker run --rm \
  -v "$PWD/test.mp4":/app/test.mp4 \
  -v "$PWD/output":/app/output \
  -e STREAM=/app/test.mp4 \
  -e OUTPUT=file \
  privacy-layer
```

### On an RTSP camera

Verified working with an Amcrest IP camera over a direct Ethernet link.
The plugin opens the stream through OpenCV's RTSP support (FFMPEG-backed):

```bash
docker run --rm \
  -v "$PWD/output":/app/output \
  -e STREAM='rtsp://admin:PASSWORD@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1' \
  -e OUTPUT=file \
  -e PUBLISH_COUNT=1 \
  -e PYWAGGLE_LOG_DIR=/app/output \
  privacy-layer
```

`subtype=1` requests the camera's sub-stream (lower resolution, lower
bitrate). Use `subtype=0` for the main stream.

URL-encode any special characters in the password (`@` &rarr; `%40`,
`:` &rarr; `%3A`, `/` &rarr; `%2F`).

### On a Sage node

```bash
sudo pluginctl build .
sudo pluginctl run --name privacy-layer \
  -e CAMERA=top \
  -e OUTPUT=upload \
  -e UPLOAD_EVERY=150 \
  -e PUBLISH_COUNT=1 \
  <image-tag>
```

The `CAMERA=` value must match a `match.id` entry in
`/run/waggle/data-config.json` on the node.

## Output

### File mode

A blurred mp4 is written to `output/<timestamp>/blurred.mp4` (or whatever
`OUT_PATH` is set to). Useful for processing pre-recorded video.

### Upload mode

One blurred JPG is encoded and shipped to Beehive every `UPLOAD_EVERY`
frames. On a Sage node, uploads land in object storage and become
queryable through the Sage data API:

```bash
curl -s -H 'Content-Type: application/json' \
  https://data.sagecontinuum.org/api/v1/query \
  -d '{"start":"-5m","filter":{"task":"privacy-layer"}}'
```

### Measurements

When `PUBLISH_COUNT=1`, the plugin publishes:

- `privacy.faces.count` &mdash; integer count of faces detected per frame
- `privacy.method` &mdash; the masking method active for the run

The plugin **does not** publish bounding boxes, face embeddings, identity
labels, demographic estimates, or any other derived attribute that could
be used for re-identification.

## Local development without a Sage node

The plugin works on any Linux box with Docker. When pywaggle's `Camera`
class fails to resolve a source &mdash; typically because
`/run/waggle/data-config.json` is missing &mdash; the code falls back to
opening the source through raw OpenCV. This lets the same image run
unchanged on a developer laptop, a plain Jetson Thor, or a fully
configured Sage Blade.

You will see this log line once on startup when the fallback activates:
pywaggle Camera rejected source (...); using cv2 fallback

It is harmless and indicates dev mode; on a real Sage node it will not
appear.

For local testing, set `PYWAGGLE_LOG_DIR` to capture published
measurements to a local `data.ndjson` file instead of routing them to a
real Beehive.

## Verified configurations

| Hardware | Source | Method | Status |
|---|---|---|---|
| Jetson Thor (arm64), Docker 29.2.1 | RTSP from Amcrest IP camera | `box` | Working |
| Jetson Thor (arm64), Docker 29.2.1 | Local mp4 file | `box`, `gaussian`, `pixelate`, `solid`, `median` | Working |

## Known caveats

- The Amcrest's RTSP authentication is sensitive to special characters
  in the password &mdash; URL-encode if needed.
- `cv2.VideoWriter` uses the `mp4v` codec by default. Some players
  require `avc1`; if the output mp4 won't play, change the FOURCC in
  `make_writer()`.
- The plugin does not currently publish a heartbeat in `none` output
  mode, so a deployed node has no observable signal that the plugin is
  alive other than the scheduler's container status. Set
  `PUBLISH_COUNT=1` if you want continuous health visibility.
- `Camera()` in older pywaggle versions does not accept HTTP URLs; the
  cv2 fallback handles them transparently.

## License

MIT