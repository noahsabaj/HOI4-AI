# Direct import since we're in the same directory
import time
import numpy as np
from fast_memory import FastEpisodicMemory

print("🧪 Testing LMDB Performance...")

# Initialize
memory = FastEpisodicMemory()

# Test single write
embedding = np.random.randn(128).astype(np.float32)
action = {'type': 'click', 'x': 100, 'y': 200}
context = {'screen': 'production', 'date': '1936'}

start = time.perf_counter()
memory.store_experience(embedding, action, 10.0, context)
single_time = (time.perf_counter() - start) * 1000

print(f"\n✅ Single write: {single_time:.3f}ms")

# Test batch write
print("\n📊 Testing batch performance...")
batch_size = 1000
embeddings = np.random.randn(batch_size, 128).astype(np.float32)
actions = [{'type': 'click', 'x': i, 'y': i} for i in range(batch_size)]
rewards = np.random.randn(batch_size) * 5
contexts = [{'step': i} for i in range(batch_size)]

start = time.perf_counter()
memory.store_batch(embeddings, actions, rewards.tolist(), contexts)
batch_time = time.perf_counter() - start

print(f"✅ Batch write ({batch_size} items): {batch_time*1000:.1f}ms")
print(f"   That's {batch_size/batch_time:.0f} writes/second!")

# Test search
print("\n🔍 Testing search performance...")
query = np.random.randn(128).astype(np.float32)

start = time.perf_counter()
results = memory.search_similar(query, k=50)
search_time = (time.perf_counter() - start) * 1000

print(f"✅ Found {len(results)} similar memories in {search_time:.2f}ms")

# Test high-reward retrieval
start = time.perf_counter()
high_reward = memory.get_high_reward_memories(min_reward=3.0, limit=10)
reward_time = (time.perf_counter() - start) * 1000

print(f"✅ Retrieved {len(high_reward)} high-reward memories in {reward_time:.2f}ms")

memory.close()
print("\n🎉 All tests passed!")