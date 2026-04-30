#ifndef KUIPER_INCLUDE_MODEL_KVCACHE_MANAGER_H_
#define KUIPER_INCLUDE_MODEL_KVCACHE_MANAGER_H_
#include <vector>
#include <queue>
#include "base/base.h"
#include "tensor/tensor.h"

namespace model {

class KVCacheManager {
 public:
  KVCacheManager(int32_t block_num, int32_t block_size, base::DeviceType device_type);

  // 分配一个物理块，返回物理块的索引
  int32_t allocate_block();

  // 释放一个物理块
  void free_block(int32_t physical_block_idx);

  // 清空所有块
  void clear();

  // 返回当前可用的空闲块数
  int32_t get_free_block_num() const;

  int32_t get_block_size() const;

 private:
  int32_t block_num_ = 0;
  int32_t block_size_ = 0;
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;

  std::queue<int32_t> free_blocks_;
};

}  // namespace model
#endif  // KUIPER_INCLUDE_MODEL_KVCACHE_MANAGER_H_
