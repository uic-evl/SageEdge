#!/usr/bin/env python3
import sys
import os

print("=" * 60)
print("LLaVA Installation Check")
print("=" * 60)

# Try to import both
try:
    import llava
    print(f"✓ llava found at: {llava.__file__}")
    llava_path = os.path.dirname(os.path.dirname(llava.__file__))
    print(f"  Package root: {llava_path}")
except ImportError:
    print("✗ llava not found")

try:
    import llavamini
    print(f"✓ llavamini found at: {llavamini.__file__}")
    llavamini_path = os.path.dirname(os.path.dirname(llavamini.__file__))
    print(f"  Package root: {llavamini_path}")
except ImportError:
    print("✗ llavamini not found")

print("\nPython path:")
print(f"  {sys.executable}")

print("\nTrying to import specific modules:")
modules = [
    "llava.model.builder",
    "llavamini.model.builder",
]

for mod in modules:
    try:
        m = __import__(mod, fromlist=[''])
        print(f"  ✓ {mod}")
    except ImportError as e:
        print(f"  ✗ {mod}: {e}")

# Check if they're the same package with different names
try:
    import llava
    import llavamini
    if llava.__file__ == llavamini.__file__:
        print("\n⚠️  llava and llavamini point to the SAME location!")
    else:
        print("\n⚠️  llava and llavamini are DIFFERENT packages!")
except ImportError:
    pass

print("=" * 60)