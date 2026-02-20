import ollama
import os
import time

print("--- 🚀 Starting Thor Vision Test ---")

# 1. Identify the image
image_file = "test.jpg"

if not os.path.exists(image_file):
    print(f"❌ Error: {image_file} not found in current directory.")
else:
    print(f"📸 Found image: {image_file}")
    
    # 2. Test Models
    for model_name in ["moondream", "gemma3"]:
        print(f"\n📡 Testing {model_name}...")
        try:
            start = time.time()
            
            # Note: stream=False makes it wait for the full answer
            response = ollama.generate(
                model=model_name,
                prompt="Describe this image in 10 words or less.",
                images=[image_file],
                stream=False
            )
            
            end = time.time()
            print(f"💬 {model_name.upper()} says: {response['response'].strip()}")
            print(f"⏱️  Time: {end - start:.2f}s")
            
        except Exception as e:
            print(f"⚠️  {model_name} failed: {e}")

print("\n--- Test Complete ---")
