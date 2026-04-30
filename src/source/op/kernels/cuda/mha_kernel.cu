#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include "mha_kernel.cuh"
#include <base/tick.h>
namespace kernel {
constexpr static int thread_num = 256;

struct MaxOp {
  __device__ __forceinline__ float operator()(const float& a, const float& b) const {
    return (a > b) ? a : b;
  }
};

__device__ void softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;

  // find max value (for numerical stability)
  // this should be FLT_MAX, not 0 !!!!
  // otherwise, the softmax may be occur nan when head_dim < 128 threads
  float max_val = tid < size ? x[tid] : -FLT_MAX;
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  using BlockReduce = cub::BlockReduce<float, thread_num>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, MaxOp());
  if (threadIdx.x == 0) {
    shared_val = max_val;
  }
  __syncthreads();
  max_val = shared_val;

  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}


#define Bc 16
#define Br 16

__global__ void flash_attention_v2_kernel(int32_t pos, int32_t seq_len, float* query,
                                            float* score_ptr, float* output, float* key_cache,
                                            float* value_cache, int32_t* block_table,
                                            int32_t kv_dim, int32_t kv_mul,
                                            int32_t head_num, int32_t head_size,
                                            int32_t layer_offset, int32_t kv_block_size) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  // NOTE: For decoding pos = seq_len - 1 is passed to pos. So actually the sequence length is pos + 1.
  int total_seqlen = pos + 1;
  float smScale = 1.f / sqrtf(float(head_size));

  // block size for K, V
  // group of row(seqlen)
  int groupSeq = (total_seqlen + Bc - 1) / Bc;
  // parallel process for V[Br, d]
  // group of column
  int groupTx = (head_size + Bc - 1) / Bc;
  int groupTy = (head_size + Br - 1) / Br;

  // load slice from global memory(HBM)
  extern __shared__ float shared_mem[];
  float* sQ = shared_mem;
  float* sK = shared_mem + Br * head_size;
  float* sV = shared_mem + 2 * Br * head_size;
  float* sO = shared_mem + 3 * Br * head_size;
  float* sQK = shared_mem + 4 * Br * head_size;
  float* sSafeE = shared_mem + 4 * Br * head_size + Br * Bc;
  float* sDenom = shared_mem + 4 * Br * head_size + 2 * Br * Bc;
  float* sMax = shared_mem + 4 * Br * head_size + 2 * Br * Bc + Br;

  // [0, Bc]
  int tx = threadIdx.x;
  // [0, Br]
  int ty = threadIdx.y;

  // blockIdx.y is the query sequence row block.
  // Since it's decoding/prefilling, we calculate one row for each token.
  // For prefill pos could be sequence length - 1, and seq_len is pos+1.
  // But in this implementation, the query size is always 1 token.
  int row = ty + blockIdx.y * blockDim.y;

  int head_offset = (head / kv_mul) * head_size;
  float* query_head = query + head * head_size;
  float* output_head = output + head * head_size;

  // load q, o, max, denom from global memory to shared memory
  // Q[Br, dim]
  for (int i = 0; i < groupTx; i++) {
    if (i * Bc + tx < head_size) {
      if (row < 1) {
        sQ[ty * head_size + i * Bc + tx] = query_head[row * head_size + i * Bc + tx];
      } else {
        sQ[ty * head_size + i * Bc + tx] = 0.f;
      }
      sO[ty * head_size + i * Bc + tx] = 0;
    }
  }

  if (tx == 0) {
    sMax[ty] = -INFINITY;
    sDenom[ty] = 0;
  }
  __syncthreads();

  // load K, V block
  // Q[Br][dim] @ K[0..seqlen.step(Bc), dim]
  // compute partial sum of O[ty][dim] each iteration
  for (int j = 0; j < groupSeq; j++) {
    // wait until previous iteration g2s done
    __syncthreads();
    
    if ((j * Bc + tx) < total_seqlen) {
      // load k, v from global memory to shared memory
      // K[seqlen, dim], V[seqlen, dim]
      int t = j * Bc + tx;
      int logical_block_idx = t / kv_block_size;
      int physical_block_idx = block_table[logical_block_idx];
      int block_offset = t % kv_block_size;
      
      float* key_head = key_cache + layer_offset + physical_block_idx * kv_block_size * kv_dim + block_offset * kv_dim + head_offset;
      float* value_head = value_cache + layer_offset + physical_block_idx * kv_block_size * kv_dim + block_offset * kv_dim + head_offset;

      for (int i = 0; i < groupTy; i++) {
        if (i * Br + ty < head_size) {
          sK[tx * head_size + i * Br + ty] = key_head[i * Br + ty];
          sV[tx * head_size + i * Br + ty] = value_head[i * Br + ty];
        }
      }
    } else { // padding for elements outside sequence length
      for (int i = 0; i < groupTy; i++) {
        if (i * Br + ty < head_size) {
          sK[tx * head_size + i * Br + ty] = 0;
          sV[tx * head_size + i * Br + ty] = 0;
        }
      }
    }

    // wait until g2s done
    __syncthreads();

    // compute qk
    float sum = 0.f;
    // result oriented: qk[y][x] from q[y] @ k[x]
    for (int i = 0; i < head_size; i++) {
      sum += sQ[ty * head_size + i] * sK[tx * head_size + i];
    }
    // sQK[Br, Bc]
    if (j * Bc + tx < total_seqlen && row < 1) {
        sQK[ty * Bc + tx] = sum * smScale;
    } else {
        sQK[ty * Bc + tx] = -INFINITY; // mask
    }

    // wait until qk done
    __syncthreads();

    // compute local max of each row of qk
    float localMax = -INFINITY;
    for (int i = 0; i < Bc; i++) {
      localMax = max(localMax, sQK[ty * Bc + i]);
    }
    __syncthreads();
    // compute the max of each row
    float newMax = max(sMax[ty], localMax);

    // compute safe e(e^{x - max}) of each qk element
    sSafeE[ty * Bc + tx] = exp(sQK[ty * Bc + tx] - newMax);
    __syncthreads();

    // accumulate local denom of each row of qk with local max
    float localDenom = 0.f;
    for (int i = 0; i < Bc; i++) {
      localDenom += sSafeE[ty * Bc + i];
    }
    __syncthreads();

    // rescale history result
    float rescaleOld = exp(sMax[ty] - newMax);
    // rescale denom
    float newDenom = sDenom[ty] * rescaleOld + localDenom;

    // NOTE:
    // QK[Br, Bc] @ V[Bc, d] = O[Br, d]
    // tx in [0, Bc], ty in [0, Br]
    // slice-Bc and each O[ty, group.x] as accumulator
    for (int i = 0; i < groupTx; i++) {
      if (i * Bc + tx < head_size) {
        // NOTE: rescale old_o(numerator only for now) once: old_nume * rescale
        sO[ty * head_size + i * Bc + tx] = (sO[ty * head_size + i * Bc + tx] * rescaleOld);
        for (int k = 0; k < Bc; k++) {
          // NOTE:
          // accumulate numerator
          // new_nume = old_nume' + local_nume (Softmax(QK)@V)
          sO[ty * head_size + i * Bc + tx] += sSafeE[ty * Bc + k] * sV[k * head_size + i * Bc + tx];
        }
      }
    }

    // update global max and denom
    if (tx == 0) {
      sMax[ty] = newMax;
      sDenom[ty] = newDenom;
    }
    __syncthreads();
  }

  // rescale O in the end
  for (int i = 0; i < groupTx; i++) {
    if (i * Bc + tx < head_size) {
      if (row < 1) {
        // copy sO[row, dim] to gO[row, dim]
        output_head[row * head_size + i * Bc + tx] = sO[ty * head_size + i * Bc + tx] / sDenom[ty];
      }
    }
  }
}

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, int32_t kv_block_size,
                   const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                   const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
                   const tensor::Tensor& value_cache_tensor, const tensor::Tensor& block_table_tensor,
                   base::DeviceType device_type, CudaConfig* config) {
  UNUSED(device_type);
  int32_t kv_block_num = (seq_len + kv_block_size - 1) / kv_block_size;
  int32_t layer_offset = layer_index * kv_block_num * kv_block_size * kv_dim;
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());

  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());
  int32_t* block_table = const_cast<int32_t*>(block_table_tensor.ptr<int32_t>());

  cudaStream_t stream = config->stream;
  
  dim3 grid(head_num);
  dim3 block(Bc, Br);
  
  // memory size: sQ, sK, sV, sO, sQK, sSafeE, sDenom, sMax
  // We allocate 256 for head_size in shared memory pointers above just as a max cap.
  // Actually shared_mem needs to be sized properly.
  // sQ[Br][head_size], sK[Bc][head_size], sV[Bc][head_size], sO[Br][head_size]
  // sQK[Br][Bc], sSafeE[Br][Bc], sDenom[Br], sMax[Br]
  int smem_size = (4 * Br * head_size + 2 * Br * Bc + 2 * Br) * sizeof(float);
  
  flash_attention_v2_kernel<<<grid, block, smem_size, stream>>>(
      pos, seq_len, query, score, output, key_cache, value_cache, block_table, kv_dim, kv_mul, head_num,
      head_size, layer_offset, kv_block_size);
}

}  // namespace kernel
