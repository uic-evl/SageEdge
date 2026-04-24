import ollama
import os
import time
from pathlib import Path

print("=" * 60)
print("🔍 AI Vision Models Caption Test")
print("=" * 60)

# Find all test images in current directory
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
test_images = [f for f in os.listdir('.') if any(f.lower().endswith(ext) for ext in image_extensions)]

if not test_images:
    print("❌ No test images found in current directory.")
    print(f"   Looking for: {', '.join(image_extensions)}")
    exit(1)

print(f"\n📸 Found {len(test_images)} test image(s): {', '.join(test_images)}")

# Define models to test
models_to_test = {
    'ollama': ['moondream', 'llama3.2-vision', 'llava'],  # Common vision models in Ollama
    'yolo': 'yolo11n'  # YOLO for object detection
}

prompt = "Describe this image in detail, including objects, people, and activities you see."

print("\n" + "=" * 60)
print("Testing Ollama Vision Models")
print("=" * 60)

# Test each image with each Ollama model
for image_file in test_images:
    print(f"\n📷 Testing with image: {image_file}")
    print("-" * 60)
    
    for model_name in models_to_test['ollama']:
        print(f"\n🤖 Model: {model_name}")
        try:
            # Check if model is available
            models_list = ollama.list()
            model_names = [m['name'].split(':')[0] for m in models_list.get('models', [])]
            
            if model_name not in model_names:
                print(f"   ⚠️  Model '{model_name}' not installed. Run: ollama pull {model_name}")
                continue
            
            start = time.time()
            
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                images=[image_file],
                stream=False
            )
            
            end = time.time()
            
            caption = response['response'].strip()
            print(f"   ✅ Caption: {caption[:200]}{'...' if len(caption) > 200 else ''}")
            print(f"   ⏱️  Time: {end - start:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

# Test YOLO (object detection)
print("\n" + "=" * 60)
print("Testing YOLO Object Detection")
print("=" * 60)

try:
    from ultralytics import YOLO
    
    for image_file in test_images:
        print(f"\n📷 Testing with image: {image_file}")
        print("-" * 60)
        print(f"🤖 Model: YOLO11n")
        
        try:
            start = time.time()
            
            # Load YOLO model
            model = YOLO('yolo11n.pt')
            
            # Run inference
            results = model(image_file, verbose=False)
            
            end = time.time()
            
            # Extract detected objects
            detected_objects = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names[class_id]
                    detected_objects.append(f"{class_name} ({confidence:.2f})")
            
            if detected_objects:
                print(f"   ✅ Detected: {', '.join(detected_objects[:10])}")
                if len(detected_objects) > 10:
                    print(f"   ... and {len(detected_objects) - 10} more objects")
            else:
                print(f"   ⚠️  No objects detected")
            
            print(f"   ⏱️  Time: {end - start:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            
except ImportError:
    print("\n⚠️  YOLO (ultralytics) not installed.")
    print("   Install with: pip install ultralytics")

print("\n" + "=" * 60)
print("✨ Test Complete")
print("=" * 60)
