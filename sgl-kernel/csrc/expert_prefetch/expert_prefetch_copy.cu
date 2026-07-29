#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/irange.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <vector>

#if !defined(USE_ROCM) && !defined(USE_MUSA)
#include <dlfcn.h>
#endif

#include "pytorch_extension_utils.h"

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
        void**, void**, size_t*, size_t, cudaMemcpyAttributes*, size_t*, size_t, size_t*, cudaStream_t);
    auto fn = reinterpret_cast<FnV12>(cuda_memcpy_batch_async_sym);
    err = fn(
        batch_dsts.data(),
        batch_srcs.data(),
        const_cast<size_t*>(batch_sizes.data()),
        num_copies,
        &attrs,
        attrs_idxs.data(),
        1,
        &fail_idx,
        stream);
  }

  if (err == cudaErrorNotSupported || err == cudaErrorCallRequiresNewerDriver) {
    return false;
  }
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "cudaMemcpyBatchAsync failed. failIdx=", fail_idx, " error=", cudaGetErrorString(err));
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
#if !defined(USE_ROCM) && !defined(USE_MUSA)
    if (!try_memcpy_batch_async(batch_dsts, batch_srcs, batch_sizes, stream)) {
      launch_memcpy_loop(batch_dsts, batch_srcs, batch_sizes, stream);
    }
#else
    launch_memcpy_loop(batch_dsts, batch_srcs, batch_sizes, stream);
#endif
    return;
  }

  at::Tensor ids_cpu = expert_ids.contiguous().to(at::kCPU, at::kLong);
  const int64_t num_ids = ids_cpu.numel();
  if (num_ids == 0) {
    return;
  }
  const int64_t* ids_ptr = ids_cpu.data_ptr<int64_t>();

  std::vector<void*> batch_dsts;
  std::vector<void*> batch_srcs;
  std::vector<size_t> batch_sizes;
  batch_dsts.reserve(srcs.size() * static_cast<size_t>(num_ids));
  batch_srcs.reserve(srcs.size() * static_cast<size_t>(num_ids));
  batch_sizes.reserve(srcs.size() * static_cast<size_t>(num_ids));

  for (const auto i : c10::irange(srcs.size())) {
    append_expert_row_copies(
        srcs[i], const_cast<at::Tensor&>(dsts[i]), ids_ptr, num_ids, batch_dsts, batch_srcs, batch_sizes);
  }

#if !defined(USE_ROCM) && !defined(USE_MUSA)
  if (!try_memcpy_batch_async(batch_dsts, batch_srcs, batch_sizes, stream)) {
    launch_memcpy_loop(batch_dsts, batch_srcs, batch_sizes, stream);
  }
#else
  launch_memcpy_loop(batch_dsts, batch_srcs, batch_sizes, stream);
#endif
}
