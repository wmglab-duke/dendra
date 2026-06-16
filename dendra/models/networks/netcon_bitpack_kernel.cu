#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kBitsPerWord = 63;
constexpr int kThreads = 256;

__global__ void zero_history_row_kernel(
    int64_t* __restrict__ packed_history,
    const int64_t* __restrict__ current_time_step,
    int64_t n_words) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_words) {
    const int64_t slot = current_time_step[0];
    packed_history[slot * n_words + i] = 0;
  }
}

__global__ void pack_source_spikes_kernel(
    const bool* __restrict__ source_spikes,
    int64_t* __restrict__ packed_history,
    const int64_t* __restrict__ current_time_step,
    int64_t n_source,
    int64_t n_words) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_source || !source_spikes[i]) {
    return;
  }
  const int64_t slot = current_time_step[0];
  const int64_t word = i / kBitsPerWord;
  const int64_t bit = i - word * kBitsPerWord;
  const unsigned long long mask = 1ULL << bit;
  auto* row = reinterpret_cast<unsigned long long*>(packed_history + slot * n_words);
  atomicOr(row + word, mask);
}

template <typename scalar_t>
__global__ void build_delivery_kernel(
    const int64_t* __restrict__ packed_history,
    const int64_t* __restrict__ current_time_step,
    const int64_t* __restrict__ delay_steps,
    const int64_t* __restrict__ conn_source_pos,
    const int64_t* __restrict__ post_idx,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ delivery_out,
    int64_t n_conn,
    int64_t max_delay_steps,
    int64_t n_words) {
  int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= n_conn) {
    return;
  }

  int64_t row = current_time_step[0] - delay_steps[e];
  row %= max_delay_steps;
  if (row < 0) {
    row += max_delay_steps;
  }

  const int64_t src = conn_source_pos[e];
  const int64_t word = src / kBitsPerWord;
  const int64_t bit = src - word * kBitsPerWord;
  const unsigned long long mask = 1ULL << bit;
  const auto* hist = reinterpret_cast<const unsigned long long*>(packed_history);
  const unsigned long long packed = hist[row * n_words + word];

  if ((packed & mask) != 0ULL) {
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
  auto stream = at::cuda::getCurrentCUDAStream();

  if (n_words > 0) {
    const int64_t zero_blocks = (n_words + kThreads - 1) / kThreads;
    zero_history_row_kernel<<<zero_blocks, kThreads, 0, stream>>>(
        packed_history.data_ptr<int64_t>(),
        current_time_step.data_ptr<int64_t>(),
        n_words);
  }
  if (n_source > 0) {
    const int64_t blocks = (n_source + kThreads - 1) / kThreads;
    pack_source_spikes_kernel<<<blocks, kThreads, 0, stream>>>(
        source_spikes.data_ptr<bool>(),
        packed_history.data_ptr<int64_t>(),
        current_time_step.data_ptr<int64_t>(),
        n_source,
        n_words);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void build_delivery_cuda(
    torch::Tensor packed_history,
    torch::Tensor current_time_step,
    torch::Tensor delay_steps,
    torch::Tensor conn_source_pos,
    torch::Tensor post_idx,
    torch::Tensor weight,
    torch::Tensor delivery_out) {
  const int64_t n_conn = delay_steps.numel();
  const int64_t max_delay_steps = packed_history.size(0);
  const int64_t n_words = packed_history.size(1);
  if (n_conn == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t blocks = (n_conn + kThreads - 1) / kThreads;

  AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "netcon_bitpack_build_delivery", [&] {
    build_delivery_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        packed_history.data_ptr<int64_t>(),
        current_time_step.data_ptr<int64_t>(),
        delay_steps.data_ptr<int64_t>(),
        conn_source_pos.data_ptr<int64_t>(),
        post_idx.data_ptr<int64_t>(),
        weight.data_ptr<scalar_t>(),
        delivery_out.data_ptr<scalar_t>(),
        n_conn,
        max_delay_steps,
        n_words);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
