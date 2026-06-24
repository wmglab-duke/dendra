#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kBitsPerWord = 63;
constexpr int kPackThreads = 64;
constexpr int kDeliveryThreads = 256;

// Pack one source-spike vector into one row of packed_history.
// One CTA owns one int64 word and writes that word exactly once.  This avoids
// the old two-kernel zero+atomicOr path:
//   1) zero current row
//   2) one thread per active source with atomicOr(row[word], bit)
// The sign bit is intentionally unused; source bits are 0..62.
__global__ void pack_source_spikes_words_kernel(
    const bool* __restrict__ source_spikes,
    int64_t* __restrict__ packed_history,
    const int64_t* __restrict__ current_time_step,
    int64_t n_source,
    int64_t n_words) {
  const int64_t word = static_cast<int64_t>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);

  unsigned long long v = 0ULL;
  const int64_t src = word * kBitsPerWord + lane;
  if (lane < kBitsPerWord && src < n_source && source_spikes[src]) {
    v = 1ULL << lane;
  }

  __shared__ unsigned long long smem[kPackThreads];
  smem[lane] = v;
  __syncthreads();

  if (lane < 32) smem[lane] |= smem[lane + 32];
  __syncthreads();
  if (lane < 16) smem[lane] |= smem[lane + 16];
  __syncthreads();
  if (lane < 8) smem[lane] |= smem[lane + 8];
  __syncthreads();
  if (lane < 4) smem[lane] |= smem[lane + 4];
  __syncthreads();
  if (lane < 2) smem[lane] |= smem[lane + 2];
  __syncthreads();
  if (lane < 1) smem[lane] |= smem[lane + 1];
  __syncthreads();

  if (lane == 0) {
    const int64_t slot = current_time_step[0];
    auto* hist = reinterpret_cast<unsigned long long*>(packed_history);
    hist[slot * n_words + word] = smem[0];
  }
}

template <typename scalar_t>
__global__ void build_delivery_kernel(
    const int64_t* __restrict__ packed_history,
    const int64_t* __restrict__ current_time_step,
    const int64_t* __restrict__ delay_steps,
    const int64_t* __restrict__ conn_word_idx,
    const int64_t* __restrict__ conn_bit_mask,
    const int64_t* __restrict__ post_idx,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ delivery_out,
    int64_t n_conn,
    int64_t max_delay_steps,
    int64_t n_words) {
  const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= n_conn) {
    return;
  }

  // inference_delay_steps is pre-clamped to >=1 and max_delay_steps is one
  // larger than the maximum representable integer delay.  Therefore one add is
  // enough to implement modulo for cur-delay.
  int64_t row = current_time_step[0] - delay_steps[e];
  if (row < 0) {
    row += max_delay_steps;
  }

  const int64_t word = conn_word_idx[e];
  const unsigned long long bit_mask = static_cast<unsigned long long>(conn_bit_mask[e]);
  const auto* hist = reinterpret_cast<const unsigned long long*>(packed_history);
  const unsigned long long packed = hist[row * n_words + word];

  if ((packed & bit_mask) != 0ULL) {
    atomicAdd(delivery_out + post_idx[e], weight[e]);
  }
}

template <typename scalar_t>
__global__ void build_delivery_uniform_kernel(
    const int64_t* __restrict__ packed_history,
    const int64_t* __restrict__ current_time_step,
    int64_t delay_step,
    const int64_t* __restrict__ conn_word_idx,
    const int64_t* __restrict__ conn_bit_mask,
    const int64_t* __restrict__ post_idx,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ delivery_out,
    int64_t n_conn,
    int64_t max_delay_steps,
    int64_t n_words) {
  const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= n_conn) {
    return;
  }

  int64_t row = current_time_step[0] - delay_step;
  if (row < 0) {
    row += max_delay_steps;
  }

  const int64_t word = conn_word_idx[e];
  const unsigned long long bit_mask = static_cast<unsigned long long>(conn_bit_mask[e]);
  const auto* hist = reinterpret_cast<const unsigned long long*>(packed_history);
  const unsigned long long packed = hist[row * n_words + word];

  if ((packed & bit_mask) != 0ULL) {
    atomicAdd(delivery_out + post_idx[e], weight[e]);
  }
}

}  // namespace

void pack_source_spikes_cuda(
    torch::Tensor source_spikes,
    torch::Tensor packed_history,
    torch::Tensor current_time_step) {
  const int64_t n_source = source_spikes.numel();
  const int64_t n_words = packed_history.size(1);
  if (n_words == 0) {
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  pack_source_spikes_words_kernel<<<n_words, kPackThreads, 0, stream>>>(
      source_spikes.data_ptr<bool>(),
      packed_history.data_ptr<int64_t>(),
      current_time_step.data_ptr<int64_t>(),
      n_source,
      n_words);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void build_delivery_cuda(
    torch::Tensor packed_history,
    torch::Tensor current_time_step,
    torch::Tensor delay_steps,
    torch::Tensor conn_word_idx,
    torch::Tensor conn_bit_mask,
    torch::Tensor post_idx,
    torch::Tensor weight,
    torch::Tensor delivery_out) {
  const int64_t n_conn = delay_steps.numel();
  const int64_t max_delay_steps = packed_history.size(0);
  const int64_t n_words = packed_history.size(1);
  if (n_conn == 0 || max_delay_steps == 0 || n_words == 0) {
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t blocks = (n_conn + kDeliveryThreads - 1) / kDeliveryThreads;

  AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "netcon_bitpack_build_delivery", [&] {
    build_delivery_kernel<scalar_t><<<blocks, kDeliveryThreads, 0, stream>>>(
        packed_history.data_ptr<int64_t>(),
        current_time_step.data_ptr<int64_t>(),
        delay_steps.data_ptr<int64_t>(),
        conn_word_idx.data_ptr<int64_t>(),
        conn_bit_mask.data_ptr<int64_t>(),
        post_idx.data_ptr<int64_t>(),
        weight.data_ptr<scalar_t>(),
        delivery_out.data_ptr<scalar_t>(),
        n_conn,
        max_delay_steps,
        n_words);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void build_delivery_uniform_cuda(
    torch::Tensor packed_history,
    torch::Tensor current_time_step,
    int64_t delay_step,
    torch::Tensor conn_word_idx,
    torch::Tensor conn_bit_mask,
    torch::Tensor post_idx,
    torch::Tensor weight,
    torch::Tensor delivery_out) {
  const int64_t n_conn = conn_word_idx.numel();
  const int64_t max_delay_steps = packed_history.size(0);
  const int64_t n_words = packed_history.size(1);
  if (n_conn == 0 || max_delay_steps == 0 || n_words == 0) {
    return;
  }

  // Preserve the same invariant as the Python-side inference-delay metadata.
  if (delay_step < 1) {
    delay_step = 1;
  }
  if (delay_step >= max_delay_steps) {
    delay_step %= max_delay_steps;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t blocks = (n_conn + kDeliveryThreads - 1) / kDeliveryThreads;

  AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "netcon_bitpack_build_delivery_uniform", [&] {
    build_delivery_uniform_kernel<scalar_t><<<blocks, kDeliveryThreads, 0, stream>>>(
        packed_history.data_ptr<int64_t>(),
        current_time_step.data_ptr<int64_t>(),
        delay_step,
        conn_word_idx.data_ptr<int64_t>(),
        conn_bit_mask.data_ptr<int64_t>(),
        post_idx.data_ptr<int64_t>(),
        weight.data_ptr<scalar_t>(),
        delivery_out.data_ptr<scalar_t>(),
        n_conn,
        max_delay_steps,
        n_words);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
