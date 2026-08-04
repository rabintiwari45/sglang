// Standalone expert_prefetch_copy for JIT install into sgl_kernel.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/irange.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <vector>

#if !defined(USE_ROCM) && !defined(USE_MUSA)
#include <dlfcn.h>
#endif

namespace {

void launch_memcpy_loop(
    const std::vector<void*>& dsts,
    const std::vector<void*>& srcs,
    const std::vector<size_t>& sizes,
    cudaStream_t stream) {
  for (size_t i = 0; i < srcs.size(); ++i) {
    C10_CUDA_CHECK(cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyHostToDevice, stream));
  }
}

#if !defined(USE_ROCM) && !defined(USE_MUSA)
bool try_memcpy_batch_async(
    const std::vector<void*>& batch_dsts,
    const std::vector<void*>& batch_srcs,
    const std::vector<size_t>& batch_sizes,
    cudaStream_t stream) {
#if !defined(CUDA_VERSION) || CUDA_VERSION < 12080
  return false;
#else
  int driver_version = 0;
  if (cudaDriverGetVersion(&driver_version) != cudaSuccess || driver_version < 12080) {
    return false;
  }

  static void* cuda_memcpy_batch_async_sym = dlsym(RTLD_DEFAULT, "cudaMemcpyBatchAsync");
  if (cuda_memcpy_batch_async_sym == nullptr) {
    return false;
  }

  static int runtime_version = 0;
  static cudaError_t runtime_version_err = cudaRuntimeGetVersion(&runtime_version);
  if (runtime_version_err != cudaSuccess) {
    return false;
  }
  static const bool use_v13_signature = runtime_version >= 13000;

  if (batch_srcs.empty()) {
    return true;
  }

  std::vector<size_t> attrs_idxs(1, 0);
  cudaMemcpyAttributes attrs{};
  const int device_id = at::cuda::current_device();
  attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  attrs.srcLocHint.type = cudaMemLocationTypeHost;
  attrs.srcLocHint.id = 0;
  attrs.dstLocHint.type = cudaMemLocationTypeDevice;
  attrs.dstLocHint.id = device_id;
  attrs.flags = 0;

  cudaError_t err;
  size_t fail_idx = std::numeric_limits<size_t>::max();
  const size_t num_copies = batch_srcs.size();
  if (use_v13_signature) {
    using FnV13 = cudaError_t (*)(
        void* const*,
        const void* const*,
        const size_t*,
        size_t,
        cudaMemcpyAttributes*,
        size_t*,
        size_t,
        cudaStream_t);
    auto fn = reinterpret_cast<FnV13>(cuda_memcpy_batch_async_sym);
    err = fn(
        batch_dsts.data(),
        reinterpret_cast<const void* const*>(batch_srcs.data()),
        batch_sizes.data(),
        num_copies,
        &attrs,
        attrs_idxs.data(),
        1,
        stream);
  } else {
    using FnV12 = cudaError_t (*)(
        void* const*,
        const void* const*,
        const size_t*,
        size_t,
        cudaMemcpyAttributes*,
        size_t*,
        size_t,
        size_t*,
        cudaStream_t);
    auto fn = reinterpret_cast<FnV12>(cuda_memcpy_batch_async_sym);
    err = fn(
        batch_dsts.data(),
        reinterpret_cast<const void* const*>(batch_srcs.data()),
        batch_sizes.data(),
        num_copies,
        &attrs,
        attrs_idxs.data(),
        1,
        &fail_idx,
        stream);
  }

  if (err != cudaSuccess) {
    // Batch API is flaky across driver/runtime combos; fall back to a
    // tight cudaMemcpyAsync loop (still much cheaper than Python/ctypes).
    (void)fail_idx;
    return false;
  }
  return true;
#endif
}
#endif

void append_expert_row_copies(
    const at::Tensor& src,
    at::Tensor& dst,
    const int64_t* expert_ids,
    int64_t num_ids,
    std::vector<void*>& batch_dsts,
    std::vector<void*>& batch_srcs,
    std::vector<size_t>& batch_sizes) {
  TORCH_CHECK(src.dim() >= 1, "expert prefetch src must be at least 1D");
  TORCH_CHECK(dst.sizes() == src.sizes(), "expert prefetch src/dst shape mismatch");
  TORCH_CHECK(src.is_cpu(), "expert prefetch src must be CPU");
  TORCH_CHECK(dst.is_cuda(), "expert prefetch dst must be CUDA");

  const int64_t src_row_stride_bytes = src.stride(0) * src.element_size();
  const int64_t dst_row_stride_bytes = dst.stride(0) * dst.element_size();
  const size_t row_bytes = static_cast<size_t>(src_row_stride_bytes);
  const char* src_base = static_cast<const char*>(src.data_ptr());
  char* dst_base = static_cast<char*>(dst.data_ptr());

  for (int64_t i = 0; i < num_ids; ++i) {
    const int64_t e = expert_ids[i];
    TORCH_CHECK(e >= 0 && e < src.size(0), "expert id out of range: ", e);
    batch_srcs.push_back(const_cast<char*>(src_base + e * src_row_stride_bytes));
    batch_dsts.push_back(dst_base + e * dst_row_stride_bytes);
    batch_sizes.push_back(row_bytes);
  }
}

}  // namespace

void expert_cache_copy(
    const std::vector<at::Tensor>& srcs,
    const std::vector<at::Tensor>& dsts,
    const at::Tensor& src_rows,
    const at::Tensor& dst_rows) {
  // Copy srcs[i][src_rows[j]] -> dsts[i][dst_rows[j]] for every tensor pair
  // and row pair, on the current stream.  Tensors may live on host (pinned)
  // or device; direction is resolved per pair (cudaMemcpyDefault / UVA).
  // Used for expert-cache hit restores (D2D) and cache inserts (D2D).
  TORCH_CHECK(srcs.size() == dsts.size(), "expert_cache_copy: src/dst list size mismatch");
  TORCH_CHECK(!srcs.empty(), "expert_cache_copy: empty tensor list");

  at::Tensor src_rows_cpu = src_rows.contiguous().to(at::kCPU, at::kLong);
  at::Tensor dst_rows_cpu = dst_rows.contiguous().to(at::kCPU, at::kLong);
  const int64_t n = src_rows_cpu.numel();
  TORCH_CHECK(
      dst_rows_cpu.numel() == n, "expert_cache_copy: src_rows/dst_rows length mismatch");
  if (n == 0) {
    return;
  }
  const int64_t* srow = src_rows_cpu.data_ptr<int64_t>();
  const int64_t* drow = dst_rows_cpu.data_ptr<int64_t>();

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  for (const auto i : c10::irange(srcs.size())) {
    const at::Tensor& src = srcs[i];
    const at::Tensor& dst = dsts[i];
    TORCH_CHECK(src.dim() >= 1 && dst.dim() >= 1, "expert_cache_copy: tensors must be >= 1D");

    const int64_t src_pitch = src.stride(0) * src.element_size();
    const int64_t dst_pitch = dst.stride(0) * dst.element_size();
    const size_t row_bytes = static_cast<size_t>(
        (src.numel() / src.size(0)) * src.element_size());
    TORCH_CHECK(
        row_bytes == static_cast<size_t>((dst.numel() / dst.size(0)) * dst.element_size()),
        "expert_cache_copy: row byte size mismatch at tensor ",
        i);

    const char* src_base = static_cast<const char*>(src.data_ptr());
    char* dst_base = static_cast<char*>(dst.data_ptr());

    for (int64_t j = 0; j < n; ++j) {
      const int64_t s = srow[j];
      const int64_t d = drow[j];
      TORCH_CHECK(s >= 0 && s < src.size(0), "expert_cache_copy: src row out of range: ", s);
      TORCH_CHECK(d >= 0 && d < dst.size(0), "expert_cache_copy: dst row out of range: ", d);
      C10_CUDA_CHECK(cudaMemcpyAsync(
          dst_base + d * dst_pitch,
          src_base + s * src_pitch,
          row_bytes,
          cudaMemcpyDefault,
          stream));
    }
  }
}

void expert_prefetch_copy(
    const std::vector<at::Tensor>& srcs,
    const std::vector<at::Tensor>& dsts,
    const at::Tensor& expert_ids,
    bool copy_all) {
  TORCH_CHECK(srcs.size() == dsts.size(), "expert_prefetch_copy: src/dst list size mismatch");
  TORCH_CHECK(!srcs.empty(), "expert_prefetch_copy: empty tensor list");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (copy_all) {
    std::vector<void*> batch_dsts;
    std::vector<void*> batch_srcs;
    std::vector<size_t> batch_sizes;
    batch_dsts.reserve(srcs.size());
    batch_srcs.reserve(srcs.size());
    batch_sizes.reserve(srcs.size());
    for (const auto i : c10::irange(srcs.size())) {
      const at::Tensor& src = srcs[i];
      at::Tensor& dst = const_cast<at::Tensor&>(dsts[i]);
      TORCH_CHECK(src.is_cpu(), "expert prefetch src must be CPU");
      TORCH_CHECK(dst.is_cuda(), "expert prefetch dst must be CUDA");
      TORCH_CHECK(src.sizes() == dst.sizes(), "expert prefetch src/dst shape mismatch");
      const size_t nbytes = static_cast<size_t>(src.nbytes());
      batch_srcs.push_back(src.data_ptr());
      batch_dsts.push_back(dst.data_ptr());
      batch_sizes.push_back(nbytes);
    }
    launch_memcpy_loop(batch_dsts, batch_srcs, batch_sizes, stream);
    return;
  }

  // Prefer a CPU long view without an extra host copy when possible.
  at::Tensor ids_cpu = expert_ids;
  if (!ids_cpu.defined() || ids_cpu.numel() == 0) {
    return;
  }
  if (!ids_cpu.device().is_cpu() || ids_cpu.scalar_type() != at::kLong || !ids_cpu.is_contiguous()) {
    ids_cpu = ids_cpu.contiguous().to(at::kCPU, at::kLong);
  }
  const int64_t num_ids = ids_cpu.numel();
  const int64_t* ids_ptr = ids_cpu.data_ptr<int64_t>();

  // Fast path: issue cudaMemcpyAsync directly (no batch API — it is unsupported
  // / invalid on many driver combos and the failed attempt is expensive).
  for (const auto i : c10::irange(srcs.size())) {
    const at::Tensor& src = srcs[i];
    at::Tensor& dst = const_cast<at::Tensor&>(dsts[i]);
    TORCH_CHECK(src.dim() >= 1, "expert prefetch src must be at least 1D");
    TORCH_CHECK(dst.sizes() == src.sizes(), "expert prefetch src/dst shape mismatch");
    TORCH_CHECK(src.is_cpu(), "expert prefetch src must be CPU");
    TORCH_CHECK(dst.is_cuda(), "expert prefetch dst must be CUDA");

    const int64_t src_row_stride_bytes = src.stride(0) * src.element_size();
    const int64_t dst_row_stride_bytes = dst.stride(0) * dst.element_size();
    const size_t row_bytes = static_cast<size_t>(src[0].numel() * src.element_size());
    const char* src_base = static_cast<const char*>(src.data_ptr());
    char* dst_base = static_cast<char*>(dst.data_ptr());

    for (int64_t j = 0; j < num_ids; ++j) {
      const int64_t e = ids_ptr[j];
      TORCH_CHECK(e >= 0 && e < src.size(0), "expert id out of range: ", e);
      C10_CUDA_CHECK(cudaMemcpyAsync(
          dst_base + e * dst_row_stride_bytes,
          src_base + e * src_row_stride_bytes,
          row_bytes,
          cudaMemcpyHostToDevice,
          stream));
    }
  }
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("expert_prefetch_copy(Tensor[] srcs, Tensor[] dsts, Tensor expert_ids, bool copy_all) -> ()");
  m.def("expert_cache_copy(Tensor[] srcs, Tensor[] dsts, Tensor src_rows, Tensor dst_rows) -> ()");
}

TORCH_LIBRARY_IMPL(sgl_kernel, CUDA, m) {
  m.impl("expert_prefetch_copy", &expert_prefetch_copy);
  m.impl("expert_cache_copy", &expert_cache_copy);
}
