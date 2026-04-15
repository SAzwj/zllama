#ifndef KUIPER_INCLUDE_MODEL_SEQUENCE_CONTEXT_H_
#define KUIPER_INCLUDE_MODEL_SEQUENCE_CONTEXT_H_
#include <vector>
#include <memory>
#include "kvcache_manager.h"

namespace model {

class SequenceContext {
 public:
  SequenceContext(std::shared_ptr<KVCacheManager> kv_cache_manager);

  // 为 pos 分配物理块，如果相应的逻辑块未分配物理块，则分配并更新 block_table
  void allocate_blocks_for_pos(int32_t pos);

  // 重置序列状态并归还已分配的物理块
  void reset();

  // 获取物理块表
  const std::vector<int32_t>& get_block_table() const;

  // 返回对应逻辑块的物理块索引
  int32_t get_physical_block_idx(int32_t logical_block_idx) const;

  std::shared_ptr<KVCacheManager> get_kv_cache_manager() const;

 private:
  std::shared_ptr<KVCacheManager> kv_cache_manager_;
  std::vector<int32_t> block_table_;
};

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_SEQUENCE_CONTEXT_H_
