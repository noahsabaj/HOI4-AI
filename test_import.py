try:
    import test_simple
    print(f"✅ Success! 2 + 3 = {test_simple.add(2, 3)}")
except ImportError as e:
    print(f"❌ Failed to import: {e}")