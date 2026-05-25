# APPLICATION LAYER

**Purpose:** Model-specific TensorRT inference implementations for 16+ vision models.

## STRUCTURE

```
application/
├── app_yolo/          # Detection (YOLOv3-v13, YOLOX)
├── app_yolo_cls/      # Classification
├── app_yolo_seg/      # Instance segmentation
├── app_yolo_pose/     # Pose estimation (17 COCO keypoints)
├── app_yolo_obb/      # Oriented bounding box
├── app_rtdetr/        # RT-DETR transformer detector
├── app_rtmo/          # MMPose RTMO pose
├── app_ppocr/         # PaddleOCR v4 (det+cls+rec pipeline)
├── app_bytetrack/     # ByteTrack multi-object tracking
├── app_laneatt/       # LaneATT lane detection
├── app_clrnet/        # CLRNet lane detection
├── app_clrernet/      # CLRerNet lane detection
├── app_depth_anything/ # Monocular depth estimation
├── cuosd/             # CUDA on-screen drawing (boxes, masks, text)
├── common/            # Shared Box struct, Face::Box
└── tools/             # DeepSORT, ZMQ streaming, pybind11 bindings
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| YOLO decode/NMS | `app_yolo/yolo_decode.cu` |
| Segmentation mask decode | `app_yolo_seg/yolo_seg_decode.cu` |
| Pose keypoint decode | `app_yolo_pose/yolo_pose_decode.cu` |
| OBB angle decode | `app_yolo_obb/yolo_obb_decode.cu` |
| RT-DETR transformer decode | `app_rtdetr/rtdetr_decode.cu` |
| ByteTrack tracking | `app_bytetrack/byte_tracker.cpp` |
| OCR det→cls→rec pipeline | `app_ppocr/ppocr.cpp` |
| CUDA drawing primitives | `cuosd/cuosd.cpp`, `cuosd_kernel.cu` |

## MODULE PATTERN

Each `app_<model>/` follows identical structure:

```
app_yolo/
├── yolo.hpp           # Interface: Infer abstract class + create_infer()
├── yolo.cpp           # Implementation: InferImpl : public Infer
├── yolo_decode.cu     # CUDA kernels: decode_kernel_invoker + nms_kernel_invoker
└── multi_gpu.hpp      # Optional multi-GPU wrapper
```

## CONVENTIONS

- **Interface Pattern**: `class Infer { virtual shared_future<BoxArray> commit(const Mat&) = 0; }`
- **Factory Pattern**: `shared_ptr<Infer> create_infer(engine_file, gpuid, conf_thresh, nms_thresh, ...)`
- **Namespace**: Each model in own namespace (`Yolo`, `YoloSeg`, `YoloPose`, `RTDETR`, `PPOCR`, etc.)
- **Base Box Format**: `left, top, right, bottom, confidence, class_label` (6 elements)
- **Extended Boxes**:
  - Segmentation: `+ shared_ptr<InstanceSegmentMap> seg`
  - Pose: `+ vector<Point3f> keypoints` (17 COCO keypoints)
  - OBB: `+ float angle` (radians)
- **Preprocessing**: `CUDAKernel::resize_bilinear_and_normalize()` + `warp_affine_bilinear()`
- **Postprocessing**: Custom CUDA kernels in `*_decode.cu` for parallel decode + NMS
- **Controller**: All use `InferController<Mat, BoxArray, ...>` for async batch processing

## ANTI-PATTERNS

- **DO NOT** skip `multi_gpu.hpp` if targeting multi-GPU deployment
- **DO NOT** modify Box element order - atomic counter batching depends on memory layout
- **DO NOT** use CPU NMS for production - `NMSMethod::FastGPU` is ~10x faster
- **DO NOT** call `commit()` synchronously - use `commit().get()` or batch with `commits()`
