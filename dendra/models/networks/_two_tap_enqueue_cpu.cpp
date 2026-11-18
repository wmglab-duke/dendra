#include <ATen/ATen.h>
#include <c10/util/bit_cast.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>

namespace {

void check_input(
    const at::Tensor& input,
    at::ScalarType dtype,
    int64_t size,
    const char* name) {
  TORCH_CHECK(
      input.device().is_cpu() && input.layout() == at::kStrided &&
          input.is_contiguous() && input.scalar_type() == dtype &&
          input.numel() == size,
      name,
      " must be a contiguous CPU tensor of the required dtype and size");
}

template <typename scalar_t, bool FromSource>
void add_tap(
    scalar_t* output,
    const scalar_t* weights,
    const scalar_t* gates,
    const scalar_t* fractions,
    const int64_t* addresses,
    const int64_t* pre_idx,
    int64_t source_count,
    int64_t count,
    int64_t ring_size,
    int64_t slot_start,
    bool lower_tap) {
  const int64_t tail_size = ring_size - slot_start;
  for (int64_t edge = 0; edge < count; ++edge) {
    const int64_t base = addresses[edge];
    TORCH_CHECK(
        static_cast<uint64_t>(base) < static_cast<uint64_t>(ring_size),
        "Two-tap address is outside the delivery ring");
    const int64_t destination =
        base < tail_size ? slot_start + base : base - tail_size;
    int64_t gate_index = edge;
    if constexpr (FromSource) {
      gate_index = pre_idx[edge];
      TORCH_CHECK_INDEX(
          static_cast<uint64_t>(gate_index) < static_cast<uint64_t>(source_count),
          "Two-tap source index is out of range");
    }
    const scalar_t weighted_gate = weights[edge] * gates[gate_index];
    const scalar_t fraction =
        lower_tap ? scalar_t(1) - fractions[edge] : fractions[edge];
    output[destination] += weighted_gate * fraction;
  }
}

template <bool InPlace, bool FromSource>
at::Tensor enqueue_two_tap(
    const at::Tensor& queue,
    const at::Tensor& slot,
    const at::Tensor& weights,
    const at::Tensor& gates,
    const at::Tensor& fraction,
    const at::Tensor& lower,
    const at::Tensor& upper,
    const at::Tensor& pre_idx) {
  const auto dtype = queue.scalar_type();
  TORCH_CHECK(
      queue.device().is_cpu() && queue.layout() == at::kStrided &&
          queue.dim() == 2 && queue.size(0) > 0 &&
          (dtype == at::kFloat || dtype == at::kDouble),
      "Two-tap queue must be a CPU float32 or float64 matrix with positive depth");
  const int64_t edges = weights.numel();
  TORCH_CHECK(
      weights.dim() == 1 && gates.dim() == 1 && fraction.dim() == 1 &&
          lower.dim() == 1 && upper.dim() == 1,
      "Two-tap edge inputs must be vectors");
  check_input(slot, at::kLong, 1, "slot");
  check_input(weights, dtype, edges, "weights");
  check_input(gates, dtype, FromSource ? gates.numel() : edges, "gates");
  const int64_t* sources = nullptr;
  if constexpr (FromSource) {
    TORCH_CHECK(pre_idx.dim() == 1, "Two-tap source indices must be a vector");
    check_input(pre_idx, at::kLong, edges, "source indices");
    sources = pre_idx.const_data_ptr<int64_t>();
  }
  check_input(fraction, dtype, edges, "fraction");
  check_input(lower, at::kLong, edges, "lower");
  check_input(upper, at::kLong, edges, "upper");

  const int64_t slot_value = slot.const_data_ptr<int64_t>()[0];
  TORCH_CHECK(
      slot_value >= 0 && slot_value < queue.size(0),
      "Two-tap slot is outside the delivery ring");
  const int64_t width = queue.size(1);
  const int64_t ring_size = queue.numel();
  TORCH_CHECK(ring_size > 0 || edges == 0, "Empty ring cannot receive edges");

  TORCH_CHECK(!InPlace || queue.is_contiguous(),
              "The private two-tap queue must be contiguous");
  auto output = InPlace ? queue : queue.clone(at::MemoryFormat::Contiguous);
  if (ring_size == 0) {
    return output;
  }
  const int64_t slot_start = slot_value * width;
  AT_DISPATCH_FLOATING_TYPES(dtype, "ordered_two_tap_enqueue_cpu", [&] {
    scalar_t* out = output.mutable_data_ptr<scalar_t>();
    std::fill_n(out + slot_start, width, scalar_t(0));
    const scalar_t* w = weights.const_data_ptr<scalar_t>();
    const scalar_t* g = gates.const_data_ptr<scalar_t>();
    const scalar_t* alpha = fraction.const_data_ptr<scalar_t>();
    // Retain the scatter sequence for collisions, including collisions
    // between the two taps.
    add_tap<scalar_t, FromSource>(
        out, w, g, alpha, lower.const_data_ptr<int64_t>(), sources, gates.numel(),
        edges, ring_size, slot_start, true);
    add_tap<scalar_t, FromSource>(
        out, w, g, alpha, upper.const_data_ptr<int64_t>(), sources, gates.numel(),
        edges, ring_size, slot_start, false);
  });
  return output;
}

template <bool InPlace>
at::Tensor ordered_two_tap_enqueue(
    const at::Tensor& queue,
    const at::Tensor& slot,
    const at::Tensor& weights,
    const at::Tensor& gates,
    const at::Tensor& fraction,
    const at::Tensor& lower,
    const at::Tensor& upper) {
  return enqueue_two_tap<InPlace, false>(
      queue, slot, weights, gates, fraction, lower, upper, {});
}

template <bool InPlace>
at::Tensor ordered_two_tap_enqueue_from_source(
    const at::Tensor& queue,
    const at::Tensor& slot,
    const at::Tensor& weights,
    const at::Tensor& gates,
    const at::Tensor& fraction,
    const at::Tensor& lower,
    const at::Tensor& upper,
    const at::Tensor& pre_idx) {
  return enqueue_two_tap<InPlace, true>(
      queue, slot, weights, gates, fraction, lower, upper, pre_idx);
}

at::Tensor delay_ring_read_row(
    const at::Tensor& queue, const at::Tensor& slot, bool advanced) {
  if (advanced) {
    c10::List<std::optional<at::Tensor>> indices;
    indices.push_back(slot);
    return at::index(queue, indices);
  }
  return queue.index_select(0, slot);
}

int64_t wrapped_tap_address(
    int64_t base, int64_t offset, int64_t ring_size, int64_t tail_size) {
  if (tail_size > 0 &&
      static_cast<uint64_t>(base) < static_cast<uint64_t>(ring_size)) {
    return base < tail_size ? offset + base : base - tail_size;
  }
  // Match add.Tensor followed by remainder.Scalar even for authored signed
  // addresses and offsets. Unsigned addition makes integer wrap defined.
  const auto bits = static_cast<uint64_t>(base) + static_cast<uint64_t>(offset);
  const auto wrapped = c10::bit_cast<int64_t>(bits) % ring_size;
  return wrapped < 0 ? wrapped + ring_size : wrapped;
}

bool has_materialized_values(const at::Tensor& tensor) {
  return !tensor.is_neg() && !tensor._is_zerotensor();
}

void check_two_tap_reads(
    const at::Tensor& ring,
    const at::Tensor& offset_tensor,
    const at::Tensor& lower,
    const at::Tensor& upper) {
  const auto dtype = ring.scalar_type();
  const auto ring_size = ring.numel();
  TORCH_CHECK(
      ring.device().is_cpu() && ring.layout() == at::kStrided &&
          ring.is_contiguous() && ring.dim() == 2 &&
          ring_size > 0 && (dtype == at::kFloat || dtype == at::kDouble),
      "Two-tap gather requires a nonempty contiguous CPU float matrix");
  TORCH_CHECK(
      offset_tensor.dim() == 1 && lower.dim() == 1 && upper.dim() == 1,
      "Two-tap gather indices must be vectors");
  const auto edges = lower.numel();
  check_input(offset_tensor, at::kLong, 1, "offset");
  check_input(lower, at::kLong, edges, "lower");
  check_input(upper, at::kLong, edges, "upper");
  TORCH_CHECK(
      has_materialized_values(ring) && has_materialized_values(offset_tensor) &&
          has_materialized_values(lower) && has_materialized_values(upper),
      "Two-tap reads require materialized inputs with resolved values");
}

std::tuple<at::Tensor, at::Tensor> delay_ring_gather_two_tap(
    const at::Tensor& ring,
    const at::Tensor& offset_tensor,
    const at::Tensor& lower,
    const at::Tensor& upper) {
  check_two_tap_reads(ring, offset_tensor, lower, upper);
  const auto dtype = ring.scalar_type();
  const auto ring_size = ring.numel();
  const auto edges = lower.numel();
  const auto offset = offset_tensor.const_data_ptr<int64_t>()[0];
  const auto tail_size = offset >= 0 && offset < ring_size ? ring_size - offset : 0;
  const auto* lower_addresses = lower.const_data_ptr<int64_t>();
  const auto* upper_addresses = upper.const_data_ptr<int64_t>();
  auto lower_values = at::empty(lower.sizes(), ring.options());
  auto upper_values = at::empty(upper.sizes(), ring.options());
  AT_DISPATCH_FLOATING_TYPES(dtype, "delay_ring_gather_two_tap_cpu", [&] {
    const auto* values = ring.const_data_ptr<scalar_t>();
    auto* lower_output = lower_values.mutable_data_ptr<scalar_t>();
    auto* upper_output = upper_values.mutable_data_ptr<scalar_t>();
    for (int64_t edge = 0; edge < edges; ++edge) {
      lower_output[edge] =
          values[wrapped_tap_address(lower_addresses[edge], offset, ring_size, tail_size)];
      upper_output[edge] =
          values[wrapped_tap_address(upper_addresses[edge], offset, ring_size, tail_size)];
    }
  });
  return {lower_values, upper_values};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> delay_ring_two_tap_vjp(
    const at::Tensor& ring,
    const at::Tensor& offset_tensor,
    const at::Tensor& weights,
    const at::Tensor& gates,
    const at::Tensor& fraction,
    const at::Tensor& lower,
    const at::Tensor& upper,
    const std::optional<at::Tensor>& source_indices) {
  check_two_tap_reads(ring, offset_tensor, lower, upper);
  const auto dtype = ring.scalar_type();
  const auto edges = lower.numel();
  const auto source_count = gates.numel();
  const bool from_source = source_indices.has_value();
  TORCH_CHECK(weights.dim() == 1 && gates.dim() == 1 && fraction.dim() == 1,
              "Two-tap VJP operands must be vectors");
  check_input(weights, dtype, edges, "weights");
  check_input(gates, dtype, from_source ? source_count : edges, "gates");
  check_input(fraction, dtype, edges, "fraction");
  TORCH_CHECK(
      has_materialized_values(weights) && has_materialized_values(gates) &&
          has_materialized_values(fraction),
      "Two-tap VJP requires materialized operands with resolved values");
  const int64_t* indices = nullptr;
  if (from_source) {
    TORCH_CHECK(source_indices->dim() == 1 && has_materialized_values(*source_indices),
                "Two-tap VJP source indices must be a materialized vector");
    check_input(*source_indices, at::kLong, edges, "source indices");
    indices = source_indices->const_data_ptr<int64_t>();
  }

  const auto ring_size = ring.numel();
  const auto offset = offset_tensor.const_data_ptr<int64_t>()[0];
  const auto tail_size = offset >= 0 && offset < ring_size ? ring_size - offset : 0;
  const auto* lower_addresses = lower.const_data_ptr<int64_t>();
  const auto* upper_addresses = upper.const_data_ptr<int64_t>();
  auto grad_weights = at::empty(weights.sizes(), ring.options());
  auto grad_gates = at::empty(gates.sizes(), ring.options());
  auto grad_fraction = at::empty(fraction.sizes(), ring.options());
  AT_DISPATCH_FLOATING_TYPES(dtype, "delay_ring_two_tap_vjp_cpu", [&] {
    const auto* values = ring.const_data_ptr<scalar_t>();
    const auto* w = weights.const_data_ptr<scalar_t>();
    const auto* g = gates.const_data_ptr<scalar_t>();
    const auto* f = fraction.const_data_ptr<scalar_t>();
    auto* dw = grad_weights.mutable_data_ptr<scalar_t>();
    auto* dg = grad_gates.mutable_data_ptr<scalar_t>();
    auto* df = grad_fraction.mutable_data_ptr<scalar_t>();
    if (from_source) {
      std::fill_n(dg, source_count, scalar_t(0));
    }

    for (int64_t edge = 0; edge < edges; ++edge) {
      auto source = edge;
      if (from_source) {
        source = indices[edge];
        TORCH_CHECK_INDEX(source >= -source_count && source < source_count,
                          "Two-tap VJP source index is out of range");
        if (source < 0) {
          source += source_count;
        }
      }
      const auto lo = values[
          wrapped_tap_address(lower_addresses[edge], offset, ring_size, tail_size)];
      const auto hi = values[
          wrapped_tap_address(upper_addresses[edge], offset, ring_size, tail_size)];
      // Keep the original products separate, including the two products in df.
      const auto dweighted = hi * f[edge] + lo * (scalar_t(1) - f[edge]);
      const auto weighted = w[edge] * g[source];
      dw[edge] = dweighted * g[source];
      df[edge] = hi * weighted - lo * weighted;
      const auto gate_gradient = dweighted * w[edge];
      if (from_source) {
        // Match index_select_backward's serial edge order and positive zero seed.
        dg[source] += gate_gradient;
      } else {
        dg[edge] = gate_gradient;
      }
    }
  });
  return {grad_weights, grad_gates, grad_fraction};
}

void ordered_two_tap_enqueue_(
    const at::Tensor& queue,
    const at::Tensor& slot,
    const at::Tensor& weights,
    const at::Tensor& gates,
    const at::Tensor& fraction,
    const at::Tensor& lower,
    const at::Tensor& upper,
    at::TensorList /* preceding_reads */) {
  // Read row operands are explicit dependencies for compiled loop scheduling.
  ordered_two_tap_enqueue<true>(queue, slot, weights, gates, fraction, lower, upper);
}

void ordered_two_tap_enqueue_from_source_(
    const at::Tensor& queue,
    const at::Tensor& slot,
    const at::Tensor& weights,
    const at::Tensor& gates,
    const at::Tensor& fraction,
    const at::Tensor& lower,
    const at::Tensor& upper,
    const at::Tensor& pre_idx,
    at::TensorList /* preceding_reads */) {
  ordered_two_tap_enqueue_from_source<true>(
      queue, slot, weights, gates, fraction, lower, upper, pre_idx);
}

void delay_ring_adjoint_update_(
    const at::Tensor& ring,
    const at::Tensor& slot,
    const at::Tensor& row,
    const at::Tensor& first,
    at::TensorList /* preceding_reads */) {
  const auto dtype = ring.scalar_type();
  TORCH_CHECK(
      ring.device().is_cpu() && ring.is_contiguous() && ring.dim() == 2 &&
          ring.size(0) > 0 && ring.size(1) > 0 &&
          (dtype == at::kFloat || dtype == at::kDouble),
      "The private cotangent ring must be a contiguous CPU float matrix");
  const auto width = ring.size(1);
  check_input(slot, at::kLong, 1, "slot");
  check_input(first, at::kBool, 1, "first step");
  const auto payload = row.contiguous();
  check_input(payload, dtype, width, "cotangent row");
  const auto depth = ring.size(0);
  auto position = slot.const_data_ptr<int64_t>()[0];
  TORCH_CHECK_INDEX(
      position >= -depth && position < depth,
      "Cotangent slot is outside the delivery ring");
  if (position < 0) {
    position += depth;
  }

  AT_DISPATCH_FLOATING_TYPES(dtype, "delay_ring_adjoint_update_cpu", [&] {
    auto* output = ring.mutable_data_ptr<scalar_t>();
    if (first.const_data_ptr<bool>()[0]) {
      // The old cotangent was read before this call. Apply its positive-zero
      // addition once. All later updates only replace the normalized row.
      for (int64_t i = 0; i < ring.numel(); ++i) {
        const auto value = output[i];
        output[i] = value == scalar_t(0) ? scalar_t(0) : value + scalar_t(0);
      }
    }
    std::copy_n(
        payload.const_data_ptr<scalar_t>(), width, output + position * width);
  });
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(dendra_native, m) {
  m.def(
      "ordered_two_tap_enqueue(Tensor queue, Tensor slot, Tensor weights, "
      "Tensor gates, Tensor fraction, Tensor lower, Tensor upper) -> Tensor");
  m.def(
      "ordered_two_tap_enqueue_from_source(Tensor queue, Tensor slot, "
      "Tensor weights, Tensor gates, Tensor fraction, Tensor lower, "
      "Tensor upper, Tensor pre_idx) -> Tensor");
  m.def(
      "ordered_two_tap_enqueue_from_source_(Tensor(a!) queue, Tensor slot, "
      "Tensor weights, Tensor gates, Tensor fraction, Tensor lower, "
      "Tensor upper, Tensor pre_idx, Tensor[] preceding_reads) -> ()");
  m.def("delay_ring_read_row(Tensor queue, Tensor slot, bool advanced) -> Tensor");
  m.def(
      "delay_ring_gather_two_tap(Tensor ring, Tensor offset, Tensor lower, "
      "Tensor upper) -> (Tensor, Tensor)");
  m.def(
      "delay_ring_two_tap_vjp(Tensor ring, Tensor offset, Tensor weights, "
      "Tensor gates, Tensor fraction, Tensor lower, Tensor upper, "
      "Tensor? source_indices=None) -> (Tensor, Tensor, Tensor)");
  m.def(
      "ordered_two_tap_enqueue_(Tensor(a!) queue, Tensor slot, Tensor weights, "
      "Tensor gates, Tensor fraction, Tensor lower, Tensor upper, "
      "Tensor[] preceding_reads) -> ()");
  m.def(
      "delay_ring_adjoint_update_(Tensor(a!) ring, Tensor slot, Tensor row, "
      "Tensor first, Tensor[] preceding_reads) -> ()");
}

TORCH_LIBRARY_IMPL(dendra_native, CPU, m) {
  m.impl("ordered_two_tap_enqueue", ordered_two_tap_enqueue<false>);
  m.impl("ordered_two_tap_enqueue_from_source",
         ordered_two_tap_enqueue_from_source<false>);
  m.impl("ordered_two_tap_enqueue_from_source_",
         ordered_two_tap_enqueue_from_source_);
  m.impl("delay_ring_read_row", delay_ring_read_row);
  m.impl("delay_ring_gather_two_tap", delay_ring_gather_two_tap);
  m.impl("delay_ring_two_tap_vjp", delay_ring_two_tap_vjp);
  m.impl("ordered_two_tap_enqueue_", ordered_two_tap_enqueue_);
  m.impl("delay_ring_adjoint_update_", delay_ring_adjoint_update_);
}
