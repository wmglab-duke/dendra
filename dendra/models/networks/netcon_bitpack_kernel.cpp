#include <torch/extension.h>

void pack_source_spikes_cuda(
    torch::Tensor source_spikes,
    torch::Tensor packed_history,
    torch::Tensor current_time_step);

void build_delivery_cuda(
    torch::Tensor packed_history,
    torch::Tensor current_time_step,
    torch::Tensor delay_steps,
    torch::Tensor conn_source_pos,
    torch::Tensor post_idx,
    torch::Tensor weight,
    torch::Tensor delivery_out);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dtype_) TORCH_CHECK(x.scalar_type() == dtype_, #x " has wrong dtype")

void pack_source_spikes(
    torch::Tensor source_spikes,
    torch::Tensor packed_history,
    torch::Tensor current_time_step) {
  CHECK_CUDA(source_spikes);
  CHECK_CUDA(packed_history);
  CHECK_CUDA(current_time_step);
  CHECK_CONTIGUOUS(source_spikes);
  CHECK_CONTIGUOUS(packed_history);
  CHECK_CONTIGUOUS(current_time_step);
  CHECK_DTYPE(source_spikes, torch::kBool);
  CHECK_DTYPE(packed_history, torch::kInt64);
  CHECK_DTYPE(current_time_step, torch::kInt64);
  TORCH_CHECK(packed_history.dim() == 2, "packed_history must be [D, n_words]");
  TORCH_CHECK(current_time_step.numel() == 1, "current_time_step must contain one scalar");
  pack_source_spikes_cuda(source_spikes, packed_history, current_time_step);
}

void build_delivery(
    torch::Tensor packed_history,
    torch::Tensor current_time_step,
    torch::Tensor delay_steps,
    torch::Tensor conn_source_pos,
    torch::Tensor post_idx,
    torch::Tensor weight,
    torch::Tensor delivery_out) {
  CHECK_CUDA(packed_history);
  CHECK_CUDA(current_time_step);
  CHECK_CUDA(delay_steps);
  CHECK_CUDA(conn_source_pos);
  CHECK_CUDA(post_idx);
  CHECK_CUDA(weight);
  CHECK_CUDA(delivery_out);
  CHECK_CONTIGUOUS(packed_history);
  CHECK_CONTIGUOUS(current_time_step);
  CHECK_CONTIGUOUS(delay_steps);
  CHECK_CONTIGUOUS(conn_source_pos);
  CHECK_CONTIGUOUS(post_idx);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(delivery_out);
  CHECK_DTYPE(packed_history, torch::kInt64);
  CHECK_DTYPE(current_time_step, torch::kInt64);
  CHECK_DTYPE(delay_steps, torch::kInt64);
  CHECK_DTYPE(conn_source_pos, torch::kInt64);
  CHECK_DTYPE(post_idx, torch::kInt64);
  TORCH_CHECK(packed_history.dim() == 2, "packed_history must be [D, n_words]");
  TORCH_CHECK(current_time_step.numel() == 1, "current_time_step must contain one scalar");
  TORCH_CHECK(delay_steps.numel() == conn_source_pos.numel(), "delay/source size mismatch");
  TORCH_CHECK(delay_steps.numel() == post_idx.numel(), "delay/post size mismatch");
  TORCH_CHECK(delay_steps.numel() == weight.numel(), "delay/weight size mismatch");
  build_delivery_cuda(
      packed_history,
      current_time_step,
      delay_steps,
      conn_source_pos,
      post_idx,
      weight,
      delivery_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_source_spikes", &pack_source_spikes, "Pack source spikes into int64 history row (CUDA)");
  m.def("build_delivery", &build_delivery, "Build dense NetCon delivery from packed source history (CUDA)");
}
