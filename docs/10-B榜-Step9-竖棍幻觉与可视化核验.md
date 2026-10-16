# B 榜 Step9：多框模型转换、"竖棍"后处理与可视化核验

> **归档说明**：本文由项目全程技术复盘按"成果主线"重组而成，对应原复盘文档的相关章节（源文件 `archive/26中国软件杯_项目全程复盘.md`）。
> **性质**：基于代码、日志、产物实测的复盘记录，**非该阶段原始源码**；文中的代码片段为**原样节选（excerpt）**，用于说明关键实现，可能省略了与本节无关的部分。

## 目的

把 Step8 最优模型 `model_v12` 转成 Paddle 静态图，在本地全量图片上生成可视化预览供人工核验，确认后打包提交。

## 实现：转换链路

```
step8/result/model_v12/train/weights/best.pt (49.6 MB)
  │  export.py: model.export(format="onnx", imgsz=640, opset=12, batch=1, simplify=False)
  ▼
step9/model/best.onnx  (101,161.8 KB = 101.16 MB)
  │  x2paddle --framework onnx --model best.onnx --save_dir x2paddle_output
  ▼
step9/model/x2paddle_output/
  ├── x2paddle_code.py            514 行 / 56.9 KB
  ├── model.pdparams              103,547,526 B (101.12 MB)
  └── inference_model/
      ├── model.pdmodel           198,251 B (193.6 KB)
      ├── model.pdiparams         103,534,304 B (101.11 MB)
      └── model.pdiparams.info    10,470 B (10.2 KB)
```

> **与 Step7 的差异**：Step7 用 `opset=11` + `simplify=True`，Step9 用 `opset=12` + `simplify=False`。Step9 的 `model.pdmodel` 是 193.6 KB（Step7 是 164.3 KB）。

> **⚠️ 一个不一致**：`step8/result/model_v12/train/weights/best.onnx` 已经存在（98.79 MB）。但 `step9/tool/export.py` 会**重新导出**并把结果复制到 `step9/model/best.onnx`（101.16 MB）。两个 ONNX 大小相差 2.4 MB，且 step8 目录里的那个**文件时间戳晚于** step9/model 里的一个。推测：`export.py` 的 `model.export()` 把新 ONNX 写回了权重目录（覆盖了旧的），然后才复制出去 —— 所以 step8 目录里那个 98.79 MB 的 ONNX 实际是 step9 导出的产物被后续操作改动过。**这个不一致无法从现存日志解释（Step9 没有日志）。**

## ⚠️⚠️ 致命问题：X2Paddle 补丁未应用到 Step9，导致 GPU 不可用

**`step9/model/x2paddle_output/x2paddle_code.py` 是 514 行的 v1 版本**，它的 DFL 切片索引是**动态的**：

```python
Constant_7_cast = paddle.cast(x=Constant_7, dtype='int32')
Mul_cast        = paddle.cast(x=Mul_output_0, dtype='int32')
Mul_1_cast      = paddle.cast(x=Mul_1_output_0, dtype='int32')
Slice   = paddle.slice(input=dfl_Reshape_1, axes=[1],
                       starts=Constant_7_cast, ends=Mul_cast)
Slice_1 = paddle.slice(input=dfl_Reshape_1, axes=[1],
                       starts=Mul_cast, ends=Mul_1_cast)
```

而**修正版**（`step6/P2/x2paddle_512_v2/`）是硬编码：

```python
Slice   = paddle.slice(input=dfl_Reshape_1_output_0, axes=[1], starts=[0], ends=[2])
Slice_1 = paddle.slice(input=dfl_Reshape_1_output_0, axes=[1], starts=[2], ends=[4])
```

**这正是文档记载的"补丁 3：DFL Slice 索引硬编码，解决 GPU 输出全零"。Step9 的产物没有这个补丁。**

**后果确认**（`step9/predict.py`）：

```python
try:
    paddle.device.set_device('gpu'); config.enable_use_gpu(1000,0)   # 1000 MB（不是 2000）
except:
    paddle.device.set_device('cpu'); config.disable_gpu(); config.enable_mkldnn()
```

**Step9 的提交包是 CPU-only 推理。** 而 Step6 的 P2 是 GPU 推理。

> **这解释了原文档从未说清的一件事**：Step6 平台 58.28 FPS 是在 **GPU** 上跑的（`enable_use_gpu(2000,0)`）。Step9 若真的提交，FPS 会是 CPU 水平（本地 eval 是 10.81 FPS）—— **远低于 20 门槛，会被强制判 0 分**。

## 实现：全量可视化核验（`tool/visualize.py`）

**三色编码规范**（实测代码）：

| 颜色 | 代码 | 含义 |
|:---|:---|:---|
| 🔴 红色粗框（thickness **3**） | `(0, 0, 255)` | **面积最大的框** = 最终竞赛输出，标注 `MAX:{conf:.2f}` |
| 🔵 蓝色细框（thickness 1） | `(255, 0, 0)` | 置信度 ≥ 0.25 的非最大框，标注 `H:{conf:.2f}` |
| 🟡 黄色细框（thickness 1） | `(0, 255, 255)` | 置信度 < 0.25 的非最大框，标注 `L:{conf:.2f}` |

**参数**：

```python
IMGSZ = 640
CONF_HIGH = 0.25          # 蓝/黄分界
CONF_SHOW = 0.01          # 低于此不绘制
STICK_RATIO_3X = 3.0
STICK_RATIO_6X = 6.0
config.disable_gpu(); config.enable_mkldnn()
config.set_cpu_math_library_num_threads(4)
```

**letterbox 的中心偏移用的是 floor**（与 step7 `predict.py` 的 `±0.1` round **不同**）：

```python
scale = min(IMGSZ/h, IMGSZ/w)
nh, nw = int(h*scale), int(w*scale)
ph, pw = (IMGSZ-nh)//2, (IMGSZ-nw)//2      # ← floor，不是 round(±0.1)
```

> **⚠️ 这是一个静默的坐标偏差**：`visualize.py` 用 floor，而 `predict.py`（step7/step9）用 `±0.1` round。当 `IMGSZ-nh` 是奇数时两者相差 1 像素。对可视化无影响，但**说明同一项目里 letterbox 实现不统一**。

**执行结果**：`step9/vis/` 生成 **2827 张**预览图（磁盘实测 2827，时间戳 07-04 23:34）。

> **⚠️ `visualize.py` 从不创建 `vis/` 目录** —— 若目录不存在，`cv2.imwrite` 会**静默失败**。（当前目录已存在，所以没暴露。）

## ⭐⭐ 核心发现："竖棍形幻觉框"与三级降级算法

这是**整个项目中最重要的一个算法发现**，也是**原文档几乎完全没有记录**的部分。

### 现象

多框模型在部分图片上会输出**细高条状（竖棍形）的幻觉框**（`h/w` 极大），且这种框**面积常常是最大的**。由于赛制要求"面积最大的火焰"，这些幻觉框会被选中并严重拉低 F1。

### 诊断证据（`step9/tool/debug_stick.py`）

```python
print(f"Total: {len(img_paths)}, stick final boxes: {len(stick_issues)} (was 196)")
#                                                                    ↑ 修复前有 196 张
```

**修复前有 196 张图的最终输出框是竖棍形**（196/2827 = **6.9%**）。

`debug_stick.py` 的诊断元数据设计：

```python
debug = f"total={len(boxes)} max_area={...:.0f} r={h/w:.2f} c={conf:.3f}"
# reason 标签：OK | FOUND_hi | FOUND_lo | BEST_of_sticks | FALLBACK_2nd | ONLY_one
# 并按 ratio 分桶统计：(3,4] (4,6] (6,10] (10,20] (20,∞)
```

### 修复算法（`step9/predict.py` 与 `visualize.py` 共享）

```python
STICK_RATIO_3X = 3.0   # h/w > 3  视为竖棍，触发降级
STICK_RATIO_6X = 6.0   # h/w > 6  视为极端，强制抛弃

def is_stick(b, threshold=STICK_RATIO_3X):
    """判断框是否为竖棍（高 > threshold * 宽）"""
    w, h = b['w'], b['h']
    return (h / w) > threshold if w > 0 else True      # w<=0 也视为竖棍

def select_box_with_stick_handling(boxes, conf_thresh=0.25):
    if not boxes:
        return None
    sorted_boxes = sorted(boxes, key=lambda b: b['w'] * b['h'], reverse=True)
    max_box = sorted_boxes[0]

    # ① 最大框不是竖棍 → 直接返回（绝大多数情况）
    if not is_stick(max_box, STICK_RATIO_3X):
        return max_box

    # ② 降级第 1 轮：在余下框中找「高置信度 + 非竖棍」
    for b in sorted_boxes[1:]:
        if b['conf'] >= conf_thresh and not is_stick(b, STICK_RATIO_3X):
            return b

    # ③ 降级第 2 轮：放宽置信度，任意非竖棍
    for b in sorted_boxes[1:]:
        if not is_stick(b, STICK_RATIO_3X):
            return b

    # ④ 降级第 3 轮：全是竖棍 → 选比例最不极端的
    best, best_ratio = max_box, max_box['h']/max_box['w'] if max_box['w']>0 else 999
    for b in sorted_boxes[1:]:
        r = b['h'] / b['w'] if b['w'] > 0 else 999
        if r < best_ratio:
            best_ratio, best = r, b
    if best_ratio <= STICK_RATIO_6X:
        return best
    if len(sorted_boxes) >= 2:
        return sorted_boxes[1]      # 彻底放弃最大框，取面积第二大
    return max_box                  # 只有一个框，无从选择
```

### 算法思想

把"面积 argmax"升级为**带形状先验（shape prior）的面积 argmax**——先验是"火焰不应是细高条"。三级降级保证**永远有输出**（不会因过滤而空 `result`）。

**这是"面积最大"这一赛制规则被模型利用（reward hacking）的典型表现**：模型发现"输出一个细长的框"能最大化面积，从而抢占输出位。**单框模型比多框模型更容易拿高分，正是因为单框模型没学会这个技巧。**

### ⚠️ 但 step9 没有日志，所以无法量化修复效果

`step9/log/` 目录**为空**。`debug_stick.py` 的输出格式是 `was 196`，但**修复后剩余多少张没有任何记录**。`bench_fps.py`（对比 Step7 `model_v4` 与 Step9 `model_v12` 的 CPU FPS）同样无日志产出。

**这违反了项目自定的规则 2，导致 Step9 的所有关键结果都不可追溯**：

- 竖棍修复后剩余多少张？**不知道**
- Step9 的本地 F1 是多少（`eval.py` 会输出）？**不知道**
- 全量可视化检出多少候选框？**不知道**
- Step9 的 CPU FPS 是多少（`bench_fps.py`）？**不知道**

## Step9 提交包

`step9/upload/submission.zip`（**84,098.1 KB = 84.10 MB**，时间戳 **07-05 07:23**）

**zip 内容清单（实测）**：

```
predict.py               5.2 KB     ← 含竖棍降级算法
model/infer_cfg.yml      0.3 KB
model/model.pdiparams  101107.7 KB  (98.74 MB)
model/model.pdmodel      193.6 KB
```

`step9/tool/pack.py` 写入的 `infer_cfg.yml`（极简）：

```yaml
arch: YOLO
label_list:
- firebig
target_size: [640, 640]
```

> **⚠️ 标签名漂移**：`infer_cfg.yml` 写 `firebig`，但 `data12/dataset.yaml` 和所有 LabelMe JSON 用的是 **`fire`**（`step8/tool/rebuild.py` 与 `pack.py` 不一致）。因为只有单类，不影响结果，但是不一致。
>
> **⚠️ `pack.py` 没有 200 MB 限制检查**（Step7 的 `pack.py` 有）。

## ⚠️ Step9 是否真的提交了？

`step9/upload/submission.zip` 时间戳 **07-05 07:23**，晚于 Step6（06-29 12:12）和 Step7（07-03 05:06）。

而 `提交/作品_可执行文件/B榜_最终提交_submission.zip` 的大小是 **82.36 MB**，与 `step6/upload/20260629_115033_P3/submission.zip`（84,337.4 KB = 82.36 MB）**完全吻合**，与 Step9 的 84.10 MB **不吻合**。

**结论**：

| 版本 | 包大小 | 是否进入最终提交 |
|:---|:---|:---|
| Step6 P2 | 82.36 MB | ✅ **是（最终提交版本）** |
| Step7 | 82.31 MB | ❌ |
| Step9 | 84.10 MB | ❌ |

**平台实测成绩只有 Step5（FPS 14.35 ❌）和 Step6（F1 0.37529 / FPS 58.28 ✅）。Step7 / Step8 / Step9 都没有平台成绩。**

> **⚠️ 勘误**：`源码清单与说明.md` §二 把 `step7/upload/20260703_050648/predict.py` 标为"**最终最好成绩版本**"—— **不准确**，最终提交的是 Step6。

## 「7.19」后期演示工程

竞赛提交后（2026-07-19）又做了一套**成果演示生成脚本**，位于 `7.19/`。

**`run_preview_v2.py`**（226 行，A 榜 + B 榜双榜推理可视化）：

| 项 | A 榜 | B 榜 |
|:---|:---|:---|
| 模型 | `7.19/model_a/`（29,313.1 KB pdiparams） | `7.19/model_b/`（101,048.5 KB pdiparams） |
| 图像源 | `A_train/Image` 随机 10 张 | `sample100` 随机 10 张 |
| 输入尺寸 | 640×640（keep_ratio=false） | **512×512 letterbox**（注意：**不是 640**） |
| 置信度阈值 | 0.3（脚本内） | 0.25 |
| 颜色 | battery=绿 `(0,255,0)`, board=橙 `(0,165,255)`, fire=红 `(0,0,255)` | firebig=黄 `(0,255,255)` |
| 推理配置 | `disable_gpu(); switch_ir_optim(False); enable_memory_optim()` | `disable_gpu(); disable_mkldnn(); switch_ir_optim(False)` |

> **注意 B 榜用的是 512 模型** —— 与 Step6 P2 一致（不是 Step7/9 的 640）。

**技术亮点（值得记录）**：

```python
def imread_safe(path):
    """cv2.imread 不支持中文路径的替代方案"""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
```

> `cv2.imread` 在 Windows 上**无法读取含中文的路径**（本项目路径含"中国软件杯"）。用 `np.frombuffer` + `cv2.imdecode` 绕过。**这个坑要记住。**

**B 榜输出分数处理**（兼容 3 种输出形状，与 predict.py 一致）：

```python
preds = out[0].T
if   preds.shape[1] >= 6: raw, scr = preds[:,:4], preds[:,4] * preds[:,5]   # 多类
elif preds.shape[1] == 5: raw, scr = preds[:,:4], preds[:,4]                # 单类
else:                     raw, scr = preds[:,:4], preds[:,4:].max(1)        # 无 obj 分支
```

**产物**（`7.19/output/` → 同步到 `提交/介绍文档/`）：

| 文件 | 大小 |
|:---|:---|
| `A榜_preview.jpg` | 1,124.7 KB（10 图 2×5 拼版） |
| `A榜_frame_*.jpg` × 10 | 132.8 ~ 214.1 KB |
| `A榜_summary.json` | 5 KB |
| `B榜_preview.jpg` | 458.4 KB |
| `B榜_sample_*.jpg` × 10 | 44.6 ~ 47.6 KB |
| `B榜_summary.json` | 2.4 KB |

**⚠️ B 榜演示的失败**：`B榜_summary.json` 中每条记录的键是 **`annotations`（GT 标注）而不是 `detections`**：

```json
{"image": "sample_084.jpg", "num_detections": 1,
 "annotations": [{"class": "firebig (GT)", "bbox": [102.0, 102.0, 150.0, 158.0]}]}
```

这是 `fix_bbox3.py` 的**兜底分支**（`else` 分支，当 `result.json` 不存在时回退到读 XML 真值）。说明 `run_preview_v2.py` 的 B 榜推理**当时未能产出结果**（模型加载或推理失败），最终 B 榜演示图是**用真值标注画的，不是模型预测**。

对照：`A榜_summary.json` 的键是正确的 `detections`，含 `confidence` 字段，值是模型真实输出（如 `{"class":"fire","confidence":0.8243,"bbox":[-3.54,3.37,417.12,1078.04]}`）。**所以 A 榜演示成功，B 榜演示退化为真值可视化。**

**`7.19/fix_bbox*.py` 三个版本**（7,401 B / 4,959 B / 8,992 B）是通过 `subprocess` 调用 `step6/result/20260629_115033_P3/predict.py` 来生成演示图的不同尝试版本，时间戳 19:08 / 19:09 / 19:11 递进（**约 1~3 分钟一版**），说明这是一个快速试错过程。

`fix_bbox3.py` 的做法值得记录：

```python
# 切到 predict.py 所在目录再运行（保证 model/ 相对路径可解析）
cmd = f'cd /d "{B_PREDICT_DIR}" && python predict.py "{infer_txt}" "{result_json}"'
result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=B_PREDICT_DIR)
# infer.txt 里写的是【相对文件名】，靠 predict.py 的 data_txt 目录拼接逻辑解析
```

## 结论

Step9 把 `model_v12` 转成 Paddle 静态图（`model.pdmodel` 193.6 KB / `model.pdiparams` 101.11 MB）并做了 2827 张全量可视化核验，发现并修复了**"竖棍形幻觉框"**这一 reward hacking 现象（修复前 196 张 = 6.9% 的最终输出框为竖棍形，用 `STICK_RATIO_3X=3.0` / `STICK_RATIO_6X=6.0` 的三级降级算法处理）。但**补丁 #3 未应用导致静态图 DFL Slice 索引仍为动态，Step9 提交包只能 CPU-only 推理**（本地 10.81 FPS，远低于 20 门槛），且 `step9/log/` 为空使修复效果、本地 F1、CPU FPS 全部不可追溯。最终提交版本仍是 Step6 的 82.36 MB 包，Step7/Step8/Step9 均无平台成绩。
