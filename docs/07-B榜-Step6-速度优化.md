# B 榜 Step6：速度优化（唯一达标的平台成绩）

> **归档说明**：本文由项目全程技术复盘按"成果主线"重组而成，对应原复盘文档的相关章节（源文件 `archive/26中国软件杯_项目全程复盘.md`）。
> **性质**：基于代码、日志、产物实测的复盘记录，**非该阶段原始源码**；文中的代码片段为**原样节选（excerpt）**，用于说明关键实现，可能省略了与本节无关的部分。

## 目的

把 FPS 从 14.35 提到 ≥20，且精度损失可接受。

## 实现：优化策略（逐项叠加、独立文件）

按实现成本递增排序，每一步产出**独立可提交的脚本 + 独立模型目录**，便于对比与回退：

```
去 NMS  →  降分辨率  →  TensorRT  →  批量推理
 P1        P2           P3           P4
```

**本地评估工具 `step6/eval.py`**（CPU only）：

```bash
python eval.py <test_dir> <data_txt> <label_dir> [imgsz=640] [log_dir]
# test_dir 必须含 model/ 子目录；version = basename(test_dir)
# 输出：detected/total, errors, GT_total, TP/FP/FN,
#       Precision, Recall, F1, total_time_s, avg_time_ms, cpu_fps
```

**`eval.py` 的匹配算法是正确的**（与 step1/step2 的跨图 bug 形成对照）：

```python
all_gt   = [[img_id, gt_box, False], ...]                       # 第三槽 = 已匹配标志
all_pred = [[img_id, box, score, -1], ...]                      # 第四槽 = 匹配到的 GT 索引
all_pred.sort(key=lambda x: x[2], reverse=True)                 # 全测试集按分数降序
for pe in all_pred:
    for gi, ge in enumerate(all_gt):
        if ge[0] != pe[0] or ge[2]:      # <<< 同图 且 未匹配
            continue
        iou = compute_iou(pe[1], ge[1])
        if iou > best_iou: best_iou, best_idx = iou, gi
    if best_iou >= 0.5 and best_idx >= 0:
        pe[3] = best_idx ; all_gt[best_idx][2] = True
```

**这是一个写对了的实现**，作者在 `step6/eval.py` 里做对了 `step1/eval.py` 里做错的事。

**⚠️ `eval.py` 的 FPS 含义**：计时只包住 `predictor.run()`（**不含预处理/后处理**），所以报告的 "CPU FPS" 是**纯前向速率**，不是端到端吞吐。`step5/predict.py` 的端到端吞吐会更低。另：`errors` 计数的图仍写入 `predictions[image_id] = []`，所以处理失败被算作**漏检**而非排除（保守但诚实的处理）。

## 结果：各版本改动与对比

| 版本 | IMGSZ | NMS | 加速器 | Batch | 本地 F1 | 本地 CPU FPS | 平均耗时 |
|:---|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **P1** | **640** | **移除** | GPU(2000MB)→CPU | 1 | **0.9849** | 7.10 | 140.8 ms |
| **P2** | **512** | 移除 | GPU→CPU | 1 | **0.9749** | 10.81 | 92.5 ms |
| **P3** | 512 | 移除 | **TRT FP32** → GPU → CPU | 1 | 0.9749 | 10.81 | 92.5 ms |
| **P4** | 512 | 移除 | TRT FP32 → GPU → CPU | **2** | 0.9749 | 10.85 | 92.2 ms |

**P2 详细检测统计**（`20260629_114050_P2_imgsz512.log`）：

```
version: P2   timestamp: 20260629_114050   imgsz: 512
images: 100   detected: 99   errors: 0
GT_total: 100   TP: 97   FP: 2   FN: 3
Precision: 0.9798   Recall: 0.9700   F1: 0.9749
total_time_s: 9.25   avg_time_ms: 92.5   cpu_fps: 10.81
```

漏检/误检定位：`sample_063`（极小火，约 26 px 宽，位于左边缘）漏检；`sample_069` 定位偏差。

**P1 的首次尝试失败**（`20260629_113213_P1.log`）：

```
version: P1   images: 100   detected: 0   errors: 100
F1: 0.0000   total_time_s: 13.46   cpu_fps: 7.43
```

12 分钟后的第二次运行（`113351`）才正常（F1=0.9849）。**原因未记录**（属于环境/路径的瞬时问题）。

**模型文件字节级差异**：

| 版本 | `model.pdmodel` | `model.pdiparams` | `model/infer_cfg.yml` |
|:---|:---|:---|:---|
| P1 | 158,386 B | **103,534,112 B** | ✅ 存在 |
| P2 | 168,162 B | **103,473,632 B** | ❌ 缺失 |
| P3 | 168,162 B | 103,473,632 B | ❌ 缺失 |
| P4 | 168,162 B | 103,473,632 B | ❌ 缺失 |

> P1 的 `model/` 与 `step5/model/` **字节完全一致**（103,534,112 / 158,386），即 **P1 复用未改的 640 模型** —— P1 是纯粹的后处理改动。
>
> P2 的 512 模型是**真正重新导出的**（参数量不同：103,473,632 vs 103,534,112）。
>
> **⚠️ 不一致**：`export_512.py` 明确会写 `infer_cfg.yml` 到 `OUT_DIR`，但 P2/P3/P4 的 `model/` 里都没有该文件（只有 P1 有）。推测目录被 `convert_model.py` 重建过。

## ⚠️ 关键技术陷阱：改 IMGSZ 必须重导出模型

**不能只改 `predict.py` 里的 `IMGSZ`**！

YOLOv8 的检测头内有**硬编码的网格数 reshape**：

| IMGSZ | 网格数（输出第 3 维） | `x2paddle_code.py` 的 reshape |
|:---|:---|:---|
| 640 | 80² + 40² + 20² = **8400** | `shape=[1, 4, 16, 8400]` |
| 512 | 64² + 32² + 16² = **5376** | `shape=[1, 4, 16, 5376]` |

只改预处理尺寸会导致 `reshape` 报错 `5376 != 8400`。

**512 的重导出全流程**（`step6/P2/export_512.py`）：

```python
onnx_path = model.export(format="onnx", imgsz=512, batch=1, simplify=True, opset=12)
paddle_dir = os.path.join(os.path.dirname(onnx_path), "paddle_model_512")
x2paddle.convert.onnx2paddle(onnx_path, paddle_dir)
# 找 inference_model/，找不到就 os.walk 全树找含 model.pdmodel 或 model.json 的目录
# 然后复制该目录里【所有】文件（比 step5/export.py 的固定三件套更稳健）
# 再自己写 infer_cfg.yml (target_size: [512,512], label_list: [firebig])
```

**`step6/P2/convert_model.py` —— `pdparams` → 静态图的抢救脚本**：

这个文件存在的原因就是 x2paddle 的产出在 Paddle 2.6.2 上不能直接用，必须重新序列化：

```python
sys.path.insert(0, r'E:\desktop\traintoolB\step6\P2\x2paddle_512')
from x2paddle_code import ONNXModel, main as x2paddle_main

paddle.disable_static()
params = paddle.load(PARAMS_PATH)
model = ONNXModel()
model.set_dict(params, use_structured_name=True)      # 按属性路径匹配，而非扁平参数名
model.eval()
dummy_input = paddle.to_tensor(np.random.randn(1, 3, 512, 512).astype('float32'))
```

然后**三级 tracing 级联**，每级都有 try/except：

1. `paddle.jit.to_static(model.forward, input_spec=[InputSpec([1,3,512,512], 'float32')])` → `paddle.jit.save(traced, MODEL_DIR/"model")`
2. 回退：`paddle.jit.TracedLayer.trace(model, inputs=[dummy_input])` → `traced.save_inference_model(MODEL_DIR)`
3. 回退：`paddle.jit.to_static(model, input_spec=[InputSpec([None,3,512,512])])` → `paddle.jit.save(...)`（**动态 batch 维**）
4. 三级全失败 → `sys.exit(1)`

**`paddle.jit.save(..., "model")` 就是产出 `model.pdmodel` + `model.pdiparams` 那一步。**

磁盘遗留物：`step6/P2/` 下同时存在 `x2paddle_512/` 和 `x2paddle_512_v2/`，以及 `best.onnx`、`best.pt`、`convert_model.py`、`export_512.py`、`predict.py` —— **至少两次转换尝试的证据**。**`_v2` 目录不被任何脚本引用**（人工操作留下的）。

## 实现：P3 / P4 的实现细节

**P3 的三级回退链**：

```python
try:   # 1. GPU + TensorRT
    config = Config(pdmodel, params); config.enable_use_gpu(2000, 0)
    config.enable_tensorrt_engine(
        workspace_size=1 << 30,          # 1 GiB
        max_batch_size=1,
        min_subgraph_size=3,
        precision_mode=Config.Precision.Float32,
        use_static=True,
        use_calib_mode=False
    )
    return create_predictor(config)
except Exception: pass
try:   # 2. 纯 GPU
    config.enable_use_gpu(2000, 0); ...
    return create_predictor(config)
except Exception: pass
# 3. CPU
config.disable_gpu(); ...
```

**TRT 参数的解释**：

- **FP32**（非 FP16/INT8）避免任何量化精度损失，且**不需要校准数据**（所以 `use_calib_mode=False`）
- **`use_static=True`** 让 Paddle 把构建好的引擎缓存到模型旁边 —— **只有首次运行付引擎构建成本**，这对限时提交是必需的
- **`min_subgraph_size=3`** 避免 TRT 接管太小的子图（收益不抵开销）

> **P3 不支持 `model.json` PIR 回退** —— 三级块里全部硬编码 `Config(pdmodel, params)`。推测是因为 TRT 对 PIR 图支持有问题。

**P4 的批量累加器**（闭包实现）：

```python
def flush_batch():
    if not batch_tensors: return
    if len(batch_tensors) == 1:
        box = process_single(...)                       # 绝不发 batch=1
        add_result(batch_img_ids[0], box)
    else:
        boxes = run_batch(...)
        if boxes is None:                               # batch 失败 → 逐图回退
            for t, info, img_id in zip(...): add_result(img_id, process_single(...))
        else:
            for img_id, box in zip(batch_img_ids, boxes): add_result(img_id, box)
    batch_tensors, batch_infos, batch_img_ids = [], [], []
```

`run_batch` 做 `np.stack(batch_tensors, axis=0)`，**任何异常都返回 `None`** 作为明确的"回退单图"信号。循环结束后调用 `flush_batch()` 处理尾部余数。

**P4 与 P1~P3 的两个差异**：

1. P4 的 `preprocess` 返回**三维 CHW 张量**（不 `expand_dims`），batch 维在累加时才加；P1~P3 在 `preprocess` 里就加了
2. P4 的 TRT `max_batch_size=BATCH_SIZE`（=2）

## 结果：Step6 平台实测（里程碑）

**提交版本：P2（去 NMS + IMGSZ=512，不含 TRT）**
**提交包**：`step6/upload/20260629_115033_P3/submission.zip`（82.36 MB）

> ⚠️ **目录名是 `P3` 但内容是 P2** —— 这是项目中最容易混淆的一处命名。

**zip 内容清单（实测）**：

```
predict.py               4.7 KB
model/infer_cfg.yml      0.3 KB
model/model.pdiparams  101048.5 KB  (98.66 MB)
model/model.pdmodel      164.2 KB
```

| 指标 | 平台结果 | 对比 Step5 | 说明 |
|:---|:---|:---|:---|
| **FPS** | **58.28** | 14.35 (**+306%**) | 4544 张 / 78 秒 |
| **F1 Score** | **0.37529** | — | 综合得分 = F1 |
| 功能正确性 | ✅ | ✅ | 零错误退出，JSON 合规 |
| 提交时间 | 2026-06-29 12:15 | — | — |

## 核心矛盾：本地 F1 = 0.9749，平台 F1 = 0.37529

**差距 2.6 倍**。根因分析：

| # | 原因 | 证据 |
|:---|:---|:---|
| 1 | **训练集伪标签占比 98.7%** | 5313 伪标签行 / 5383 总样本；只有 70 张真实监督 |
| 2 | **`sample100` ≠ 平台测试分布** | `sample100` 场景简单、火焰明显；平台 4544 张含无火/小火/复杂场景 |
| 3 | **IMGSZ=512 加剧小目标漏检** | 分辨率降 36%，小火焰特征丢失（本地 640→512 已掉 1 个点） |
| 4 | **模型学到"大/亮火焰"偏置** | 伪标签每图只取面积最大框训练 ⇒ 模型被训练成"永远输出最大区域" |
| 5 | **去 NMS 的副作用** | 移除 NMS 意味着重叠框全部保留，面积 argmax 可能选中一个 NMS 本会压掉的错误框。**（理论上对单框无影响，但实际在 4544 张上的行为从未被验证）** |

> **关键判断**：精度问题根因在**训练数据质量**，不在 Step6 的优化手段（去 NMS 理论上零精度影响，降 512 本地只掉 1 个点）。

## Step6 的工程价值

尽管 F1 很低，Step6 解决了**硬门槛问题**（FPS 58.28 ≫ 20），使后续所有精度探索成为"改善"而非"合规"。同时确立了精度评估的残酷现实：

> **本地 100 张不可用于精度判断，只能做功能验证。真实指标只有平台能给。**

**⚠️ 但项目的后续步骤（Step7/8/9）都违反了这个刚学到的教训。**

## 结论

Step6 以"去 NMS + IMGSZ 512（重新导出 5376 网格模型）"的 P2 方案，把平台 FPS 从 14.35 提到 **58.28**（+306%），成为**全项目唯一速度与功能双达标的平台成绩**（F1 0.37529、零错误退出，提交包 82.36 MB，目录名 `20260629_115033_P3` 但内容是 P2）。P3/P4（TRT FP32、batch=2）在本地无增益。同时暴露出本地 F1 0.9749 与平台 0.37529 相差 2.6 倍的根因在于训练数据（伪标签占 98.7%、模型学到"最大区域"偏置），而非优化手段。
