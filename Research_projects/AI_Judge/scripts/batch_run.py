import json
from pathlib import Path
from run_baseline import run

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

in_dir = Path(".")
out_dir = Path("outputs/batch")
out_dir.mkdir(parents=True, exist_ok=True)

images = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
print("Found images:", len(images))

for p in images:
    result = run(str(p))
    out_path = out_dir / f"{p.stem}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("Saved", out_path.name, "| dets:", len(result.get("detections", [])))
