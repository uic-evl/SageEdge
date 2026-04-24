# 🌐 **Privacy Filter Performance Benchmarking**

### *Measuring How Anonymization Techniques Impact Edge AI on SAGE Nodes*

---

## 📘 **Overview**

This educational module teaches students how privacy-preserving computer vision techniques affect real-time performance on SAGE edge devices (Thor, Orin, etc.).
Students implement several anonymization filters—including Gaussian blur, box blur, median blur, and pixelation—then benchmark their impact on:

* **FPS (frames per second)**
* **CPU usage**
* **GPU usage**
* **Memory consumption**

This activity blends computer vision fundamentals, responsible AI practices, and performance engineering. By deploying pipelines directly on real edge hardware, students learn how anonymization choices influence accuracy, speed, and resource constraints in real-world deployments.

---

## 🎯 **Learning Objectives**

Students will be able to:

* Describe how privacy layers integrate into a vision pipeline.
* Measure system-level performance (FPS, CPU/GPU %, RAM).
* Compare the computational cost of different anonymization filters.
* Understand trade-offs between privacy strength and real-time performance.
* Deploy and test pipelines on SAGE devices such as Thor or Orin.
* Explain why privacy-preserving processing is essential in urban sensing.

---

## 🧠 **Prerequisites**

Students should be comfortable with:

* Python programming
* Object detection (YOLO models)
* Basic Linux command line
* Running containers on SAGE nodes

Knowledge of object tracking (BYTETrack, YOLO tracker) is helpful but optional.

---

# 🎥 **Why a Tracker Is Needed for Privacy Filters**

Applying blur to raw detector bounding boxes leads to:

* jittering
* box flicker
* missed anonymization when detections drop
* unstable blur strength

A **tracker** (BYTETrack or YOLO’s built-in tracking) assigns **persistent IDs** and provides smooth bounding box trajectories across frames.

This makes anonymization:

* stable
* consistent
* reliable in crowds
* realistic for deployed SAGE nodes

Trackers add minimal compute overhead and dramatically improve blur quality.

---

# 📦 **Pipeline Architecture**

```
              ┌──────────────────┐
              │   Video Stream   │
              └─────────┬────────┘
                        │
                        ▼
              ┌──────────────────┐
              │    Detector      │
              │    (YOLOv8)      │
              └─────────┬────────┘
                        │ raw bounding boxes
                        ▼
              ┌──────────────────┐
              │     Tracker      │
              │ (BYTETrack/YOLO) │
              └─────────┬────────┘
                        │ stable boxes + IDs
                        ▼
              ┌──────────────────┐
              │  Privacy Layer   │
              │ (Blur / Pixelate)│
              └─────────┬────────┘
                        │ anonymized frame
                        ▼
              ┌──────────────────┐
              │     Output       │
              └──────────────────┘
```

---

# 🧪 **Activity Structure**

## **1. Establish Baseline Performance (No Privacy Filter)**

Students first run the detection + tracking pipeline with **no anonymization** and record:

* Average FPS
* CPU usage
* GPU usage
* Memory usage

This becomes the baseline comparison point.

---

## **2. Implement Privacy Filters**

Students then test and compare at least three anonymization functions:

* **Gaussian blur**
* **Box (average) blur**
* **Median blur**
* **Pixelation**

Each filter operates on the tracked bounding boxes so the blur stays aligned with each person.

---

## **3. Benchmark Each Method**

For each filter, students collect runtime metrics:

| Metric   | Description                             |
| -------- | --------------------------------------- |
| FPS      | real-time throughput                    |
| CPU %    | total CPU usage                         |
| GPU %    | GPU workload (tegrastats or nvidia-smi) |
| Memory % | RAM usage                               |
| Visual   | blur quality, privacy strength          |

Each run produces a CSV for later analysis.

---

## **4. Visualize and Compare Performance**

Students summarize results in a table like:

| Filter Type   | Avg FPS | CPU % | GPU % | Memory | Notes |
| ------------- | ------: | ----: | ----: | -----: | ----- |
| Baseline      |         |       |       |        |       |
| Gaussian Blur |         |       |       |        |       |
| Box Blur      |         |       |       |        |       |
| Median Blur   |         |       |       |        |       |
| Pixelation    |         |       |       |        |       |

Reflection prompts:

* Which filter was fastest?
* Which provided strongest privacy?
* Which had highest GPU cost?
* Are any methods unsuitable for edge devices?

Students generate plots using the provided analysis script (`analyze_metrics.py`).

---

# ▶️ **How to Run the Pipeline**

All commands assume you are inside the repository directory.

---

## **1. Run with Default Filter**

```bash
python3 main.py --filter pixelate
```

---

## **2. Available Filters**

```bash
python3 main.py --filter gaussian
python3 main.py --filter box
python3 main.py --filter median
python3 main.py --filter pixelate
```

---

## **3. Use a Video File**

```bash
python3 main.py --filter gaussian --source video.mp4
```

---

## **4. Run on SAGE (Headless Mode)**

Thor and Orin do not support GUI windows:

```bash
python3 main.py --filter box --no-display
```

---

## **5. Save CSV Metrics**

```bash
python3 main.py --filter median --csv metrics_median.csv
```

---

## **6. Save Output Video**

```bash
python3 main.py --filter pixelate --save-video output.mp4
```

---

# 🐳 **Running in Docker on Thor / Orin**

Example:

```bash
docker run --gpus all --rm \
  -v /path/to/data:/data \
  -v /path/to/output:/output \
  privacy-lab:latest \
  python main.py \
    --filter pixelate \
    --source /data/walking_test.mp4 \
    --no-display \
    --csv /output/metrics_pixelate.csv \
    --save-video /output/pixelate_out.mp4
```

A helper script `run_all.sh` is included to run all filters sequentially.

---

# 📊 **Visualizing Results**

After you generate CSV files, run:

```bash
python analyze_metrics.py --output-dir ./output
```

This produces:

* `fps_over_time.png`
* `mean_fps_by_filter.png`
* `cpu_usage_over_time.png`
* `gpu_usage_over_time.png`

These help students compare performance across filters.



