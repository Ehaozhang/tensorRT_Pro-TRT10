# TENSORRT CORE INFERENCE ENGINE

Core TensorRT infrastructure: inference engine, ONNX parser, CUDA utilities, custom plugins for high-performance deployment.

## STRUCTURE

```
tensorRT/
├── common/              # Shared utilities
│   ├── trt_tensor.hpp       # Tensor class with MixMemory (CPU pinned + GPU)
│   ├── preprocess_kernel.cu # CUDA kernels: resize, warp_affine, normalize
│   ├── infer_controller.hpp # Async pipeline template InferController<>
│   ├── cuda_tools.hpp       # checkCudaRuntime, AutoDevice RAII
│   ├── monopoly_allocator.hpp # Thread-safe tensor pool with timeout
│   └── ilogger.hpp          # INFO, INFOE, INFOF macros
├── infer/               # Abstract inference interface
│   └── trt_infer.hpp        # TRT::Infer: forward(), input(), output()
├── builder/             # Engine compilation
│   └── trt_builder.cpp      # TRT::compile(): ONNX→engine, FP32/FP16/INT8
├── onnx/                # ONNX protobuf definitions (auto-generated)
├── onnx_parser/         # ONNX→TRT graph conversion
└── onnxplugin/          # Custom plugin system
    ├── onnxplugin.hpp       # TRTPlugin base, SetupPlugin/RegisterPlugin macros
    └── plugins/             # LayerNorm, HSwish, HSigmoid, ScatterND, DCNv2
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Tensor abstraction | `common/trt_tensor.hpp` |
| Memory management | `common/trt_tensor.hpp:MixMemory` |
| Preprocessing kernels | `common/preprocess_kernel.cu` |
| Normalization config | `common/preprocess_kernel.cuh:Norm` |
| Async inference pipeline | `common/infer_controller.hpp` |
| Thread-safe tensor pool | `common/monopoly_allocator.hpp` |
| Engine compilation | `builder/trt_builder.cpp` |
| Inference interface | `infer/trt_infer.hpp:TRT::Infer` |
| Plugin base class | `onnxplugin/onnxplugin.hpp:TRTPlugin` |
| Plugin registration | `onnxplugin/onnxplugin.hpp:SetupPlugin/RegisterPlugin` |

## KEY CLASSES

### TRT::Tensor
N-dimensional tensor with lazy CPU/GPU synchronization. Wraps `MixMemory` for dual-buffer storage.

### TRT::MixMemory
Dual-buffer memory manager. `cpu_` = pinned (page-locked), `gpu_` = device. Lazy allocation on first `cpu()` or `gpu()` call.

### TRT::Infer
Abstract interface for engine execution. Methods: `forward()`, `input()`, `output()`, `tensor(name)`, `synchronize()`.

### InferController<Input, Output, StartParam, JobAdditional>
Async job queue with worker thread. `commit()` returns `std::shared_future<Output>`. Uses `MonopolyAllocator` for tensor pooling.

### CUDAKernel::Norm
Normalization config. `NormType::MeanStd` (mean/std), `NormType::AlphaBeta` (alpha/beta). `ChannelType::Invert` for RGB→BGR.

### ONNXPlugin::TRTPlugin
Base class for custom TensorRT plugins. Override `enqueue()` for GPU kernel execution. Use `SetupPlugin` and `RegisterPlugin` macros.

## CONVENTIONS

- **Memory**: Use `MixMemory` or `TRT::Tensor`, never raw `cudaMalloc`
- **RAII**: `AutoDevice` for GPU switching, `MixMemory` for memory lifecycle
- **Streams**: `tensor->set_stream(stream, owner=true/false)` for async execution
- **Error Handling**: `checkCudaRuntime()`, `checkCudaKernel()`, `Assert()` macros
- **Logging**: `INFO()`, `INFOE()` (error), `INFOF()` (fatal) with file:line output
- **Plugin Macros**: `SetupPlugin(class_)` + `RegisterPlugin(class_)` for TRT plugin registration
- **Normalization**: `Norm::mean_std(mean, std, alpha)` or `Norm::alpha_beta(alpha, beta)`

## ANTI-PATTERNS

- **DO NOT** edit `onnx/onnx-ml.pb.*` files - auto-generated protobuf, edits will be lost
- **DO NOT** use raw `cudaMalloc`/`cudaFree` - use `MixMemory::gpu()` or `TRT::Tensor`
- **DO NOT** skip stream synchronization - preprocessing must complete before `forward()`
- **DO NOT** call `to_gpu()`/`to_cpu()` unnecessarily - lazy sync, only when needed
- **DO NOT** ignore `checkCudaKernel()` - wraps kernel launch with error detection

## UNIQUE PATTERNS

- **Pinned Memory**: `MixMemory::cpu()` returns page-locked memory for faster CPU→GPU transfers
- **Lazy Allocation**: Memory allocated only on first `cpu(size)` or `gpu(size)` call
- **Monopoly Allocator**: Thread-safe tensor pool with timeout, prevents allocation overhead in loops
- **Affine Precomputation**: `i2d`/`d2i` matrices computed once, used in CPU preprocess + GPU decode
- **Plugin Serialization**: `LayerConfig::serialize()`/`deserialize()` for engine persistence
