#include "model/kvcache_manager.h"
#include <glog/logging.h>

namespace model {

KVCacheManager::KVCacheManager(int32_t block_num, int32_t block_size, base::DeviceType device_type)
    : block_num_(block_num), block_size_(block_size), device_type_(device_type) {
  clear();
}

int32_t KVCacheManager::allocate_block() {
  if (free_blocks_.empty()) {
    LOG(FATAL) << "No free blocks available in KV cache manager.";
    return -1;
  }
  int32_t block_idx = free_blocks_.front();
  free_blocks_.pop();
  return block_idx;
}

void KVCacheManager::free_block(int32_t physical_block_idx) {
  free_blocks_.push(physical_block_idx);
}

void KVCacheManager::clear() {
  while (!free_blocks_.empty()) {
    free_blocks_.pop();
  }
  for (int32_t i = 0; i < block_num_; ++i) {
    free_blocks_.push(i);
  }
}

int32_t KVCacheManager::get_free_block_num() const {
  return free_blocks_.size();
}

int32_t KVCacheManager::get_block_size() const {
  return block_size_;
}

}  // namespace model
