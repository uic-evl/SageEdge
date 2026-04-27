# **Science**

This application is a **privacy-preserving image and video capture tool** for the Sage edge environment. It detects human faces in real time and applies a configurable masking method — Gaussian blur, box blur, median blur, pixelation, or a full block-out — directly on the edge device, *before* any frames are saved to disk or transmitted to Beehive.

The motivating problem is general. Cameras are useful sensors for an enormous range of research questions: weather and sky observation, traffic and infrastructure monitoring, wildlife behavior, vegetation and phenology, construction and built-environment change, public-space usage, air-quality estimation from visibility, and many more. Almost every outdoor camera deployment incidentally captures people. The faces of those bystanders — who never consented to being recorded — become a liability for the researcher: ethical, legal (GDPR, BIPA, IRB protocol), and infrastructural. The conventional workaround of recording everything and blurring later moves the privacy problem rather than solving it; the unredacted frames still exist somewhere, in transit and at rest.

This plugin solves the problem **at the source**. Faces are detected and masked on the node itself, in the same process that captured the frame, before anything is persisted or transmitted. The original pixels never touch storage and never cross the network. What ships to Beehive (or to disk) is already anonymized. The plugin is camera-agnostic, application-agnostic, and produces output suitable as a feedstock for any downstream analysis a researcher wants to run.

Beyond the privacy guarantee, the plugin doubles as an educational tool for students learning about edge computing. By exposing detection thresholds, masking methods, and detection-resolution downscaling as runtime parameters, learners can directly observe the trade-offs that real edge systems make: detection accuracy versus latency, visual quality versus compute cost, privacy strength versus the legibility of remaining data.

Representative use cases include:

* any computer-vision research deployment where bystanders may appear in frame
* sky, weather, cloud, and atmospheric observation cameras in populated areas
* wildlife and ecology cameras near human-trafficked spaces
* infrastructure and built-environment monitoring (construction, traffic, pedestrian areas)
* IRB-reviewed studies in public spaces where face-level data must be excluded
* compliance-sensitive deployments in jurisdictions with biometric or facial-recognition restrictions
* responsible AI@Edge research and the ethics of distributed sensor networks
* benchmarking lightweight face-detection models on constrained hardware

---

# **AI@Edge**

The application uses **YuNet**, a tiny millisecond-level face detector loaded through OpenCV's DNN module. It runs on CPU and requires no PyTorch or GPU runtime, which makes it a good fit for the constrained, heterogeneous hardware found across Sage nodes.

Workflow:

1. A frame is captured from a Waggle camera, video file, or RTSP/HTTP stream via pywaggle's `Camera` interface.
2. The frame is optionally downscaled (e.g., 1080p → 640px width) for detection only — full-resolution pixels are preserved for the blur step.
3. YuNet detects every face in the frame and returns bounding boxes scaled back to original coordinates.
4. Each box is padded outward (default 15%) to cover hairlines, ears, and detection wobble.
5. The selected masking method is applied to each padded region: box blur, Gaussian blur, median blur, pixelation, or a solid block-out.
6. The blurred frame is either discarded, written to a local mp4, or — at a configurable interval — uploaded as a snapshot to Beehive.
7. Optionally, the number of faces detected per frame is published as a Waggle measurement, so researchers can quantify human presence without ever seeing the underlying people. This signal is opt-in; the default is silent operation.

Running the workload entirely at the edge is what makes the privacy guarantee meaningful: there is no point in the pipeline at which an unblurred frame exists outside the node's RAM.

---

# **Model Capabilities**

**YuNet (face_detection_yunet_2023mar)**

* ~300 KB ONNX model, no PyTorch or CUDA dependency
* Designed for edge deployment by its authors
* WIDER FACE benchmark: AP 0.887 (easy) / 0.871 (medium) / 0.768 (hard)
* Runs at hundreds of FPS on commodity CPU; comfortably real-time on Jetson, Thor, and Pi-class nodes
* Independently validated at ~6.8 ms/frame (~147 FPS) on a laptop CPU in published face-blur work

**Masking methods**

* `box` — default; cheap averaging filter, visually close to Gaussian at matching strength
* `gaussian` — smooth and natural-looking, but the slowest per-frame operation in the pipeline
* `median` — posterized "oil painting" effect that preserves face-region edges
* `pixelate` — classic mosaic censor, the most visually recognizable as a privacy effect
* `solid` — full black mask, the strongest privacy guarantee

---

# **Ontology**

When telemetry is enabled (`publish_count=1`), the plugin publishes one Waggle measurement per processed frame:

* `privacy.faces.count` — integer count of faces detected (and therefore masked) in the frame, timestamped with the frame's capture time

At plugin startup it also publishes:

* `privacy.method` — the masking method active for this run (string)

By default the plugin runs silently, producing no telemetry at all — purely a privacy filter on the camera stream. Researchers opt in to the face-count signal when they want to quantify human activity without seeing the people.

Optional uploads (when `output=upload` or `output=both`) produce standard Waggle `upload` records pointing to blurred JPG snapshots in Sage object storage. These uploads contain no identifiable faces by construction; the masking is applied before the file is written.

Notably absent by design: this plugin does **not** publish bounding-box coordinates, face embeddings, identity labels, demographic estimates, or any other derived attribute that could function as a re-identification vector. The face count is the only quantitative signal the plugin can be configured to export.