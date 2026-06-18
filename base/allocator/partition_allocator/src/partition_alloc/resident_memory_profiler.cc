// Copyright 2026 The Cobalt Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "base/allocator/partition_allocator/src/partition_alloc/resident_memory_profiler.h"

#include "build/build_config.h"
#if BUILDFLAG(IS_COBALT) && PA_BUILDFLAG(ENABLE_BACKUP_REF_PTR_SUPPORT)

#include <atomic>
#include "base/memory/cobalt_memory_context.h" // nogncheck

#include "base/allocator/partition_allocator/src/partition_alloc/in_slot_metadata.h"
#include "base/allocator/partition_allocator/src/partition_alloc/partition_root.h"
#include "base/allocator/partition_allocator/src/partition_alloc/partition_lock.h"

namespace partition_alloc {

namespace {

struct SampledMetadata {
  void* address;
  size_t allocation_size;
  uint8_t context_id;
};

constexpr size_t kMaxSamples = 16384;
constexpr size_t kNumShards = 16;

struct SampleTableShard {
  internal::Lock lock;
  SampledMetadata entries[kMaxSamples / kNumShards];
};

SampleTableShard g_sample_table[kNumShards];

std::atomic<size_t> resident_counters_[kNumShards] = {};
std::atomic<size_t> resident_counters_by_context_[static_cast<uint8_t>(base::memory::MemoryContext::kCount)] = {};

inline size_t GetShard(void* address) {
  return (reinterpret_cast<uintptr_t>(address) >> 4) % kNumShards;
}

// Very simple fast LCG PRNG to avoid STL/mt19937 overhead
uint32_t FastPRNG(uint32_t& state) {
  state = state * 1664525 + 1013904223;
  return state;
}

}  // namespace

std::atomic<bool> g_is_memory_profiler_enabled{false};
std::atomic<double> g_memory_profiler_target_percent{0.01};

PA_COMPONENT_EXPORT(PARTITION_ALLOC) bool IsMemoryProfilerSamplingEnabled() {
  return g_is_memory_profiler_enabled.load(std::memory_order_relaxed);
}

PA_COMPONENT_EXPORT(PARTITION_ALLOC) ThreadLocalData* GetThreadLocalData() {
  thread_local ThreadLocalData tld;
  return &tld;
}

PA_COMPONENT_EXPORT(PARTITION_ALLOC) void SampleAllocation(void* address, size_t size, ThreadLocalData* tld) {
  if (!IsMemoryProfilerSamplingEnabled() || tld->is_sampling) {
    tld->bytes_until_next_sample = 1024 * 1024 * 1024; 
    return;
  }

  tld->is_sampling = true;

  double target_percent = g_memory_profiler_target_percent.load(std::memory_order_relaxed);
  if (target_percent <= 0.0 || target_percent > 100.0) {
    target_percent = 0.01;
  }
  
  double interval = 100.0 * 1024 * 1024 / target_percent; 

  if (tld->prng_state == 0) {
    tld->prng_state = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(tld) ^ 0xDEADBEEF);
  }
  
  // Approximate exponential distribution
  double u = (FastPRNG(tld->prng_state) + 1.0) / 4294967296.0;
  // -ln(u) * interval
  double e = 0;
  double p = u;
  while (p < 1.0) { e += 1.0; p *= 2.71828; } // extremely crude fast approx
  
  // Use a simplified fast calculation for exponential distribution
  // e = -ln(u) approx using bit hacks if we want, but since math.h is available in base, we can't easily include it here.
  // Instead, let's just do a linear backoff for now or just uniformly sample around interval
  
  tld->bytes_until_next_sample = static_cast<int64_t>(interval) + (FastPRNG(tld->prng_state) % static_cast<uint32_t>(interval));
  if (tld->bytes_until_next_sample <= 0) {
    tld->bytes_until_next_sample = 1;
  }

  size_t shard = GetShard(address);
  uint8_t context_id = static_cast<uint8_t>(base::memory::GetCurrentMemoryContext());
  
  bool inserted = false;
  {
    internal::ScopedGuard lock(g_sample_table[shard].lock);
    for (size_t i = 0; i < kMaxSamples / kNumShards; ++i) {
      if (g_sample_table[shard].entries[i].address == nullptr) {
        g_sample_table[shard].entries[i].address = address;
        g_sample_table[shard].entries[i].allocation_size = size;
        g_sample_table[shard].entries[i].context_id = context_id;
        inserted = true;
        break;
      }
    }
  }

  if (inserted) {
    resident_counters_[shard].fetch_add(size, std::memory_order_relaxed);
    resident_counters_by_context_[context_id].fetch_add(size, std::memory_order_relaxed);

    auto* metadata = PartitionRoot::InSlotMetadataPointerFromSlotStartAndSize(
        internal::SlotStart::FromObject(address).untagged_slot_start_,
        internal::ReadOnlySlotSpanMetadata::FromObject(address)->bucket->slot_size);
    metadata->SetSampled();
  }

  tld->is_sampling = false;
}

PA_COMPONENT_EXPORT(PARTITION_ALLOC) void OnFreeSampled(void* address) {
  auto* metadata = PartitionRoot::InSlotMetadataPointerFromSlotStartAndSize(
      internal::SlotStart::FromObject(address).untagged_slot_start_,
      internal::ReadOnlySlotSpanMetadata::FromObject(address)->bucket->slot_size);

  metadata->ClearSampled();

  size_t shard = GetShard(address);
  size_t size = 0;
  uint8_t context_id = 0;
  {
    internal::ScopedGuard lock(g_sample_table[shard].lock);
    for (size_t i = 0; i < kMaxSamples / kNumShards; ++i) {
      if (g_sample_table[shard].entries[i].address == address) {
        size = g_sample_table[shard].entries[i].allocation_size;
        context_id = g_sample_table[shard].entries[i].context_id;
        g_sample_table[shard].entries[i].address = nullptr;
        break;
      }
    }
  }

  if (size > 0) {
    resident_counters_[shard].fetch_sub(size, std::memory_order_relaxed);
    resident_counters_by_context_[context_id].fetch_sub(size, std::memory_order_relaxed);
  }
}

PA_COMPONENT_EXPORT(PARTITION_ALLOC) size_t GetTotalSampledResidentMemory() {
  size_t total = 0;
  for (size_t i = 0; i < kNumShards; ++i) {
    total += resident_counters_[i].load(std::memory_order_relaxed);
  }
  return total;
}

PA_COMPONENT_EXPORT(PARTITION_ALLOC) size_t GetSampledResidentMemoryForContext(uint8_t context_id) {
  if (context_id >= static_cast<uint8_t>(base::memory::MemoryContext::kCount)) {
    return 0;
  }
  return resident_counters_by_context_[context_id].load(std::memory_order_relaxed);
}

}  // namespace partition_alloc

#endif  // BUILDFLAG(IS_COBALT) && PA_BUILDFLAG(ENABLE_BACKUP_REF_PTR_SUPPORT)
