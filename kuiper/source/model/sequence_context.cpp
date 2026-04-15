#include "model/sequence_context.h"
#include <glog/logging.h>

namespace model {

SequenceContext::SequenceContext(std::shared_ptr<KVCacheManager> kv_cache_manager)
    : kv_cache_manager_(kv_cache_manager) {}

void SequenceContext::allocate_blocks_for_pos(int32_t pos) {
  int32_t logical_block_idx = pos / kv_cache_manager_->get_block_size();
  if (logical_block_idx >= block_table_.size()) {
    block_table_.resize(logical_block_idx + 1, -1);
  }
  if (block_table_[logical_block_idx] == -1) {
    block_table_[logical_block_idx] = kv_cache_manager_->allocate_block();
  }
}

void SequenceContext::reset() {
  for (int32_t block_idx : block_table_) {
    if (block_idx >= 0) {
      kv_cache_manager_->free_block(block_idx);
    }
  }
  block_table_.clear();
}

const std::vector<int32_t>& SequenceContext::get_block_table() const {
  return block_table_;
}

int32_t SequenceContext::get_physical_block_idx(int32_t logical_block_idx) const {
  if (logical_block_idx < 0 || logical_block_idx >= block_table_.size()) {
    LOG(FATAL) << "Logical block index " << logical_block_idx << " is out of bounds.";
    return -1;
  }
  return block_table_[logical_block_idx];
}

std::shared_ptr<KVCacheManager> SequenceContext::get_kv_cache_manager() const {
  return kv_cache_manager_;
}

}  // namespace model
