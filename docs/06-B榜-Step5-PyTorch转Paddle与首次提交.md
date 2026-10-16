# B 榜 Step5：PyTorch→Paddle 跨框架转换与首次平台提交

> **归档说明**：本文由项目全程技术复盘按"成果主线"重组而成，对应原复盘文档的相关章节（源文件 `archive/26中国软件杯_项目全程复盘.md`）。
> **性质**：基于代码、日志、产物实测的复盘记录，**非该阶段原始源码**；文中的代码片段为**原样节选（excerpt）**，用于说明关键实现，可能省略了与本节无关的部分。

## 目的

比赛强制要求 PaddlePaddle 静态图（`.pdmodel` + `.pdiparams`），而训练用的是 PyTorch/Ultralytics YOLOv8m。Step5 要打通**跨框架转换链路**并完成第一次提交。

## 实现：转换全链路

```
best.pt (YOLOv8m, nc=1, 49.6 MB)
  │  ① model.export(format="onnx", imgsz=640, batch=1, simplify=True, opset=12)
  ▼
best.onnx (98.8 MB, 单输入 [1,3,640,640], 单输出 [1,5,8400])
  │  ② X2Paddle 1.6.0（26csresult 环境）
  ▼
x2paddle_code.py (58 KB) + model.pdparams (103 MB)
  │  ③ paddle.jit.save(Paddle 2.6.2, CPU)
  ▼
model.pdmodel (158,386 B ≈ 155 KB) + model.pdiparams (103,534,112 B ≈ 98.7 MB)
```

**输出形状含义**：`[1, 5, 8400]` 中 5 = `[cx, cy, w, h, confidence]`（单类，故无类别分数向量，只有 1 个 conf），8400 = 3 个检测头的候选框总数（80²+40²+20² = 6400+1600+400 = 8400）。

**`export.py` 的双路径策略**：

- **路径 1**：`model.export(format="paddle", imgsz=640)` —— Ultralytics 直出（**失败，不再使用**）
- **路径 2**：`model.export(format="onnx", imgsz=640, batch=1, simplify=True, opset=12)` → `x2paddle.convert.onnx2paddle(onnx_path, paddle_dir)`

路径 2 后会在产出目录里找 `inference_model/`，并只复制 `["model.pdmodel", "model.pdiparams", "infer_cfg.yml"]`。

**`generate_infer_cfg()` 兜底模板**（`infer_cfg.yml` 找不到时合成）：

```yaml
mode: paddle
draw_threshold: 0.5
metric: COCO
use_dynamic_shape: false
arch: YOLO
min_subgraph_size: 3
Preprocess:
- interp: 2
  keep_ratio: false
  target_size: [640, 640]
  type: Resize
- mean: [0.0, 0.0, 0.0]
  norm_type: none
  std:  [1.0, 1.0, 1.0]
  type: NormalizeImage
- type: Permute
label_list:
- firebig
```

> **注意**：这是**纯元数据文件**。`step5/predict.py` 完全自己做预处理，**从不读 `infer_cfg.yml`**。它的存在只是为了满足竞赛打包/校验的期望。

`export.py` 的环境 hack：

```python
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')   # 否则 emoji/中文日志会让 Windows 控制台崩溃
```

## ⚠️⚠️ 版本混乱的 X2Paddle 产物（重要发现）

项目里存在 **5 个** `x2paddle_code.py`，分属 **两种互不相同的版本**，这正是原文档里"三个补丁"描述的来源与混乱之处。

| 文件 | 行数 | `paddle.slice` 的 starts/ends | `paddle.cast` |
|:---|:---:|:---|:---|
| `step5/tool/x2paddle_output/` | 514 | **动态**：`starts=Constant_7_cast, ends=Mul_cast` | 10 个（含 **4 个 float32 cast** + int32） |
| **`step5/tool/x2paddle_v2_output/`** | 504 | **硬编码**：`starts=[0], ends=[2]` / `starts=[2], ends=[4]` | **0 个** |
| `step6/P2/x2paddle_512/` | 491 | **硬编码**：同上 | **0 个** |
| **`step6/P2/x2paddle_512_v2/`** | 491 | **硬编码**：同上 | **0 个** |
| `step9/model/x2paddle_output/` | 514 | **动态**：`starts=Constant_7_cast, ends=Mul_cast` | 5 个（1 int64 + 4 int32，**无 float32**） |

**真实的三个补丁（修正版）**：

| # | 补丁 | 应用于 | 证据 |
|:---|:---|:---|:---|
| **1** | **MaxPool dilation 断言放宽** → `x2paddle/op_mapper/onnx2paddle/opset_legacy.py` ~L1980，允许 `dilation ∈ {None, [1], [1,1]}` | X2Paddle 源码（两个环境） | 文档记录；不可从产物直接验证，但转换能成功说明必须 |
| **2** | **int64/float32 类型提升** → 算术运算前显式 `paddle.cast(..., 'float32')` | **只在 `step5/tool/x2paddle_output/`（v1，514 行）** | 该文件 L485-488 有 4 个 float32 cast：<br>`paddle.add(x=paddle.cast(Gather, 'float32'), y=paddle.cast(Constant_8, 'float32'))`<br>`paddle.divide(x=Add, y=paddle.cast(Constant_9, 'float32'))`<br>`paddle.multiply(x=Div, y=paddle.cast(Constant_10, 'float32'))`<br>`paddle.multiply(x=Div, y=paddle.cast(Constant_11, 'float32'))` |
| **3** | **DFL Slice 索引硬编码** → `starts=[0], ends=[2]` / `starts=[2], ends=[4]` | **`x2paddle_v2_output/`、`x2paddle_512/`、`x2paddle_512_v2/`** | 这三个文件都是硬编码整数（见上表） |

**⚠️ 关键修正**：原文档把补丁 #2 和 #3 描述成"对生成的 `x2paddle_code.py` 约 L485-488 / L482-494 的修改，且在 26csresult 环境生效"。实测发现：

- **两个补丁是不同版本文件上的不同改动**，不是同一个文件的两处修改
- **`step9/model/x2paddle_output/`（Step9 的产物）既有 float32 cast 也没有硬编码 slice** —— 但它有 int32 的 slice 索引 cast，与 v1（514 行）几乎一致
- **补丁 #2（float32 cast）在 Step6 和 Step9 的所有产物里都不存在**

> **这意味着**：原文档声称"26csresult 的 X2Paddle 已打源补丁，自动生成硬编码 starts/ends"—— 但 Step9 的产物（同样是 26csresult 环境、同样 514 行版本）**生成了动态索引**。所以**"自动生成硬编码"的说法不成立**，硬编码是**手工编辑**的结果。

## ⚠️ DFL Slice 索引问题的真实风险（比文档描述更严重）

从 `step9/model/x2paddle_output/x2paddle_code.py` 的真实代码：

```python
# L483
Shape_output_0 = paddle.cast(x=Shape_output_0, dtype='int64')
# L485-488
Add_output_0   = paddle.add(x=paddle.cast(Gather_output_0, 'float32'),
                            y=paddle.cast(Constant_8, 'float32'))
Div_output_0   = paddle.divide(x=Add_output_0, y=paddle.cast(Constant_9, 'float32'))
Mul_output_0   = paddle.multiply(x=Div_output_0, y=paddle.cast(Constant_10, 'float32'))
Mul_1_output_0 = paddle.multiply(x=Div_output_0, y=paddle.cast(Constant_11, 'float32'))
# L489-494
Constant_7_cast = paddle.cast(x=Constant_7, dtype='int32')
Mul_cast        = paddle.cast(x=Mul_output_0, dtype='int32')
Mul_cast        = paddle.cast(x=Mul_output_0, dtype='int32')       # 重复
Mul_1_cast      = paddle.cast(x=Mul_1_output_0, dtype='int32')
# L491/L494
Slice   = paddle.slice(input=dfl_Reshape_1, axes=[1],
                       starts=Constant_7_cast, ends=Mul_cast)
Slice_1 = paddle.slice(input=dfl_Reshape_1, axes=[1],
                       starts=Mul_cast, ends=Mul_1_cast)
```

其中 `Constant_6..11` 是**从 `model.pdparams` 加载的 `int64` 参数**，`Constant_14` 不在参数文件里（所以被硬编码为 `paddle.full(..., fill_value=2.0)`）。

**风险**：如果那批 `Constant_6..11` 参数在某个转换路径里缺失，`create_parameter(..., Constant(0.0), dtype='int64')` 会让 start=0 / end=0，产生**空切片**。这是一个潜在的静默失效模式。

而硬编码版（Step6 的产物）把这行变成：

```python
Slice   = paddle.slice(input=dfl_Reshape_1_output_0, axes=[1], starts=[0], ends=[2])
Slice_1 = paddle.slice(input=dfl_Reshape_1_output_0, axes=[1], starts=[2], ends=[4])
```

彻底消除了这个风险，也让模型可在 GPU 上运行。

**⚠️❗ Step9 用回了未修补的动态索引版本** —— 这直接解释了为什么 `step9/predict.py` 的 GPU 分支写的是 `enable_use_gpu(1000, 0)`（1000 MB，不是 step6 的 2000 MB）却仍然走 CPU：

```python
# step9/predict.py
try:
    paddle.device.set_device('gpu'); config.enable_use_gpu(1000,0)
except:
    paddle.device.set_device('cpu'); config.disable_gpu(); config.enable_mkldnn()
```

**结论**：**Step9 的提交包是 CPU-only 推理**。它的 FPS 会远低于 Step6 —— 而 Step9 的 `upload/submission.zip`（84.10 MB，07-05 07:23）**没有留下任何 FPS 记录**。

## 实现：`predict.py` 设计（与官方模板的差异）

| 维度 | 官方模板 | 本项目 |
|:---|:---|:---|
| 预处理 | PaddleDetection 管线 | **自实现 YOLOv8 letterbox + BGR→RGB + /255** |
| 模型 I/O | 3 输入（image / im_shape / scale_factor） | **单输入 `[1,3,640,640]`，单输出 `[1,5,8400]`** |
| 后处理 | PaddleDetection 内置 NMS | **自实现浮点 NMS（IoU=0.45）+ 面积最大框选择** |
| GPU 策略 | 固定 GPU | **GPU 优先（2000MB, dev 0），异常自动回退 CPU** |
| PaddleDetection 依赖 | 必需 | **完全不需要** |

**关键常量**：

| 常量 | Step5 | Step6 P1 | Step6 P2~P4 | Step9 |
|:---|:---:|:---:|:---:|:---:|
| `CONF_THRESH` | 0.25 | 0.25 | 0.25 | 0.25 |
| `IOU_THRESH` | 0.45 | —（去 NMS） | —（去 NMS） | —（去 NMS） |
| `IMGSZ` | 640 | 640 | **512** | 640 |
| `enable_use_gpu` 显存 | 2000 MB | 2000 MB | 2000 MB | **1000 MB** |

**letterbox 的精确实现**（全项目统一，含 Ultralytics 的 ±0.1 非对称取整技巧）：

```python
r = min(new_shape[0]/shape[1], new_shape[1]/shape[0])
new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))
dw = (new_shape[0]-new_unpad[0])/2 ; dh = (new_shape[1]-new_unpad[1])/2
img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
top, bottom = int(round(dh-0.1)), int(round(dh+0.1))     # ← ±0.1 保证 padding 总和精确
left, right = int(round(dw-0.1)), int(round(dw+0.1))
img = cv2.copyMakeBorder(..., value=(114,114,114))
```

**Step5 的 NMS（手写纯 NumPy）**：

```python
order = scores.argsort()[::-1]
while order.size > 0:
    i = order[0]; keep.append(i)
    ... 计算 box i 与 order[1:] 的 IoU ...
    order = order[np.where(iou <= iou_threshold)[0] + 1]   # +1 正确映射回去掉的首元素
```

注释说明了理由：**"纯浮点 NMS，保证坐标精度"** —— 为避免任何库 NMS 里的 int 转换破坏亚像素坐标，影响竞赛的 IoU 评分。

**后处理的三种输出形状分支**（全项目统一）：

```python
preds = output[0].transpose(1, 0)      # [1, N, 8400] → (8400, N)
if   preds.shape[1] >= 6: bbox_raw, score = preds[:, :4], preds[:, 4] * preds[:, 5]
elif preds.shape[1] == 5: bbox_raw, score = preds[:, :4], preds[:, 4]
else:                     bbox_raw, score = preds[:, :4], preds[:, 4:].max(axis=1)
```

`>= 6` 分支假设 **YOLOv5/v8 风格 `4 + 1(obj) + nc` 头**（单类即 6 列）。`== 5` 与 `else` 是防御性回退。**这就是同一段 `postprocess` 在 P1~P4 和 `step6/eval.py` 里逐字重复的原因。**

**最终选框**：

```python
areas = (x2-x1) * (y2-y1)
best  = argmax(areas)          # ⭐ 按「面积」不是「置信度」
```

> **关键设计**：赛制要求"面积最大的火焰"，所以选框依据是 `w*h` 而非 `conf`。这是与"常规检测取最高分"最本质的区别，也是 Step9"竖棍幻觉"问题的根源。

**鲁棒性设计**：

- `data_txt` 中的相对路径自动拼接 `data_txt` 所在目录（竞赛从任意 CWD 调用，这很重要）
- 单张图异常不中断全流程（累计 `errors`，只打印前 5 条，**只有全部失败才 `exit(1)`**）—— "部分成功也是成功"的策略
- 兼容旧格式 `.pdmodel` 与 PIR 格式 `.json`
- `export.py` 的 stdout UTF-8 重配置

## 结果（首次平台实测）

| 测试场景 | 结果 | 说明 |
|:---|:---|:---|
| 本地 CPU 精度（30 张） | ✅ 30/30 检出 | 297 ms/张，FPS 3.4 |
| 本地 GPU（Paddle 2.6.2） | ⚠️ **输出全零** | Paddle 2.6.2 + CUDA 11.8 不支持 RTX 5060 Ti（CC 12.0） |
| 端到端 `python predict.py` | ✅ 退出码 0，JSON 合法 | CPU 模式（自动回退） |
| **平台功能正确性** | ✅ 无报错 | 通过 |
| **平台 FPS** | ❌ **14.35** | 4544 张 / ~317 秒 |
| 平台 F1 | 不适用（速度不达标，综合分强制 0） | — |

**提交包**：`step5/upload/20260628_222513/submission.zip`，82.36 MB。

> **结论**：功能完整、输出正确，但**差 39.5% 的速度**（需 317 → ≤227 秒）。同时暴露了一个致命工程约束：**本地无法用 GPU 复现平台推理**，只能靠平台反馈迭代。

## 结论

Step5 打通了 `best.pt → ONNX → X2Paddle → paddle.jit.save` 的静态图链路并产出可提交包（82.36 MB），平台功能校验通过但 **FPS 仅 14.35**，因未达 20 门槛而综合分强制为 0。更关键的是发现"三个 X2Paddle 补丁"实为**不同版本文件上的不同改动**（补丁 #2 仅在 `step5/tool/x2paddle_output/`，补丁 #3 硬编码仅在 `_v2` 系），且 Step9 复用未修补的动态索引版本导致只能 CPU 推理；同时本地 GPU 因 RTX 5060 Ti（CC 12.0）不被 Paddle 2.6.2 + CUDA 11.8 支持而输出全零，本地无法复现平台推理。
