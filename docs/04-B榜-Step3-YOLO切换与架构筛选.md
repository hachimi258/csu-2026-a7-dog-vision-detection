# B 榜 Step3：切换到 Ultralytics YOLO 与架构筛选

> **归档说明**：本文由项目全程技术复盘按"成果主线"重组而成，对应原复盘文档的相关章节（源文件 `archive/26中国软件杯_项目全程复盘.md`）。
> **性质**：基于代码、日志、产物实测的复盘记录，**非该阶段原始源码**；文中的代码片段为**原样节选（excerpt）**，用于说明关键实现，可能省略了与本节无关的部分。

## 目的

Step1/Step2 用的 Faster R-CNN 训练慢（10 轮约 1.5 小时）、推理慢。比赛有 FPS ≥ 20 硬门槛，必须换成实时检测器。Step3 完成**框架切换**（Ultralytics YOLO）和**架构选型**。这是全项目最重要的转折点。

## ⚠️ 第一次尝试的致命 bug（`train1.py`）

**现象**：清单 5383 行，但训练时只有 **1558 张唯一图片**。

**根因**：

```python
img_name = os.path.basename(img_path)
dst_img  = os.path.join(YOLO_TRAIN_IMG, img_name)
if not os.path.exists(dst_img):
    shutil.copy2(img_path, dst_img)
```

因为 `pseudo_mixed_optimal.txt` 故意把同一张图列了 **3 次**（`thr_095×3`），而目标文件名只用 basename，所以第 2、3 次命中 `os.path.exists` 被**静默跳过**。标签同理。

**磁盘实测（这是被证明的，不是推断）**：

| 路径 | 文件数 |
|:---|:---|
| `step3/yolo_dataset/train/images` | **1558** |
| `step3/yolo_dataset/train/labels` | **1558** |
| 期望（清单） | 5383 |

实际 = 70 真实 + (thr_095 ∪ thr_080 的唯一 stem 数) = 70 + **1488** = **1558** ✅（`thr_080 ⊇ thr_095`，所以并集是 1488 个 stem）

> **Step2 整个重采样假设在框架切换边界上被静默丢弃了**，没有任何报错或警告。这就是 `train1_1.py` 存在的直接原因。
>
> 雪上加霜：`train1.py` 里的 `random.shuffle(all_train)` **没有设种子**，所以哪次重复"赢得"basename 竞争是不确定的——同一脚本两次运行的 1558 张子集可能不同。

**其他 bug**：

- **模型名拼错**：`candidate_models = ["yolov10m.pt", "yolov11m.pt", "yolov8m.pt", "yolov11s.pt", "yolov8n.pt"]`。正确名字是 `yolo11m.pt` / `yolo11s.pt`（无 "v"）。两者都在 `YOLO(name)` 抛异常并被 `except Exception: return None` 吞掉，所以 **5 个候选里只有 3 个真训练**（`yolov10m` / `yolov8m` / `yolov8n`）。
- **`fps_estimate` 是编造的字符串**，不是测量值：`">35 (RTX5060实测基准: YOLOv8m=42 FPS)"`。它被原样写进 `summary.txt` 当成结果。**任何从该字段派生的 FPS 论述都没有依据。**
- 无 `firebig` 框的图**完全不写标签文件**（`if box is not None` 没有 else 分支）。Ultralytics 把缺失标签文件视为"无目标"，语义上凑巧一致，但**无法区分"无火"与"转换失败"**。

## 修正方案（`train1_1.py` / `train1_fixed`）

```python
if os.path.exists(YOLO_DATA_DIR):
    shutil.rmtree(YOLO_DATA_DIR)                 # 干净重建，换用新目录名
for idx, (img_path, xml_path) in enumerate(all_train):
    base = os.path.splitext(os.path.basename(img_path))[0]
    ext  = os.path.splitext(img_path)[1]
    new_name = f"{base}_{idx:05d}{ext}"          # 例：frame_001_00001.jpg
    dst_img  = os.path.join(YOLO_TRAIN_IMG, new_name)
    dst_lbl  = os.path.join(YOLO_TRAIN_LBL, f"{base}_{idx:05d}.txt")
```

用 5 位索引枚举让 5383 个条目每个都是唯一文件 ⇒ **三份 thr_095 副本物理上都存在**。同时：

- 换用**新目录 `yolo_dataset_fixed`**，避免 Ultralytics 旧缓存干扰
- `shutil.rmtree` 保证干净起点
- 为无目标图片**写空标签文件**（语义正确的 YOLO 编码）
- 模型名修正为 `["yolo11m.pt", "yolov10m.pt", "yolov8m.pt"]`（3 个最有希望的）

**磁盘实测验证修正成功**：

| 路径 | 文件数 | 期望 |
|:---|:---|:---|
| `yolo_dataset_fixed/train/images` | **5383** | 70 + 5313 = 5383 ✅ |
| `yolo_dataset_fixed/train/labels` | **5383** | ✅ |
| `yolo_dataset_fixed/val/images` | **30** | ✅ |
| `yolo_dataset_fixed/val/labels` | **30** | ✅ |

**实际文件命名**（磁盘抽样）：

```
frame_000000_03746.txt     ← 索引是打乱后训练列表中的位置
frame_000001_04071.txt
frame_000003_00955.txt     ← 同名不同索引 = 同一张图的 3 份副本
frame_000003_03199.txt
frame_000003_03710.txt
```

## 参数：训练超参数（`train1.py` / `train1_1.py` / `train1_2.py` 完全一致）

| 参数 | 值 |
|:---|:---|
| `epochs` | **3** |
| `lr0` | **1e-4**（注释："遵循 Step2 结论：lr=1e-4, 小轮数"） |
| `batch` | **16**（OOM → 8） |
| `imgsz` | **640** |
| `device` | **0** |
| `workers` | **4** |
| `patience` | **0**（禁用早停，跑满 3 轮） |
| `optimizer` | **`'SGD'`** |
| `weight_decay` | **0.0005** |
| `momentum` | **0.937** |
| `seed` | **42** |
| 选择指标 | `best.pt` 上 `model.val(split='val')` 的 **`val.box.map50`** |
| 参数统计 | `sum(p.numel() for p in model.model.parameters())/1e6` |

**标签转换**（`parse_xml_max_box`）：

```python
for obj in root.findall("object"):
    if name != "firebig": continue              # 硬过滤类名
    area = (xmax-xmin)*(ymax-ymin)
    if area > best_area: best_area, best_box = area, (xmin,ymin,xmax,ymax)
xc = (xmin+xmax)/2.0/img_width ; yc = (ymin+ymax)/2.0/img_height
w  = (xmax-xmin)/img_width     ; h  = (ymax-ymin)/img_height
xc,yc,w,h = clip(0,1) each
return (0, xc, yc, w, h)                        # class_id = 0
```

- **硬过滤类名字符串 `"firebig"`** —— 其他类名整条跳过
- **只保留面积最大的 `firebig` 框** —— "每图一框"规则在此被**第二次**强制，多火图被当单目标训练
- 图像尺寸来自 `PIL.Image.open(img_path).size` —— **不是** XML 的 `<size>`。所以这条路径是分辨率正确的（对比 `gen_labels.py` 的硬编码 640×480）
- 6 位小数格式

**数据集 YAML**：

```yaml
path: E:/desktop/traintoolB/step3\yolo_dataset
train: train/images
val: val/images
nc: 1
names: ['firebig']
```

（路径里 `/` 与 `\` 混用，是 Windows `os.path.join` 在 f-string 里的产物。Ultralytics 能容忍。）

## 结果

**未加权（有 bug）版本**（`20260626_120441_train1.log`）：

| 模型 | mAP50 | 参数量 |
|:---|:---|:---|
| yolov8m | 0.8766 | 25.9M |
| yolov10m | 0.8027 | 16.5M |
| yolov8n | 0.7160 | 3.0M |
| yolov11m / yolov11s | — | 权重名写错，加载失败 |

**加权修正版（最终）**（`20260626_165912_train1_1.log` + `20260627_023957_train1_2.log`）：

| 排名 | 模型 | mAP50 | 参数量 | 结论 |
|:---:|:---|:---:|:---:|:---|
| 🥇 | **YOLOv8m** | **0.8886** | 25.8M | 精度冠军，进入 Step4 |
| 🥈 | YOLO11m | 0.8848 | 20.0M | 轻量备选（参数少 22%） |
| 🥉 | YOLOv10m | 0.7313 | 15.3M | **淘汰**（架构不适配，精度断层） |

> **重要发现**：加权修正后 YOLOv8m 从 0.8766 → **0.8886（+1.2 个点）**，**证明 Step2 的 `thr_095×3 + thr_080×1` 配方在 YOLO 上同样有效**。而 YOLOv10m 反而从 0.8027 掉到 0.7313，说明 v10 的架构对过采样数据更敏感。

## `train1_2.py` 的续训机制（值得学习的工程技巧）

1. 若 `yolo_dataset_fixed/dataset.yaml` 缺失则中止（提示先跑 `train1_1.py`）
2. `result_dirs = glob(step3/result/20*)`；`latest_result = max(result_dirs)` —— **在 `YYYYmmdd_HHMMSS` 固定宽度命名上做字符串 max 就是时间序**（一个巧妙的技巧，但强依赖命名格式）
3. 按磁盘上已存在的文件做三路决策：
   - `weights/best.pt` 存在 → **跳过训练**，只 `model.val()` 并报告
   - 否则 `weights/last.pt` 存在 → `resume = True`，`model.train(resume=True, ...)`
   - 都不存在 → 全新 `model.train(...)`
4. **OOM 处理很周到**：捕获含 `"out of memory"` 的 `RuntimeError` → 用 `batch=8` 重试；**若之前在 resume 状态，则放弃 resume 从头开始**（因为无法在续训时改 batch）
5. 写 `summary_continue.txt`（不覆盖原始 `summary.txt`）

**⚠️ 但 `resume=True` 时 `project`/`name` 参数是无效的** —— Ultralytics 会从 checkpoint 恢复原始训练参数，忽略这两个覆盖项。代码注释声称它们"确保保存位置正确"，实际是 no-op。另外 `resume=True` 也会**恢复原始 `epochs`**，无法用它延长训练。

## 结论

Step3 完成了从 Faster R-CNN 到 Ultralytics YOLO 的框架切换：首次尝试因 basename 冲突把 5383 行清单静默压成 1558 张唯一图，`train1_1.py` 用 `{base}_{idx:05d}` 唯一命名 + `rmtree` 干净重建修复为 5383 张。加权修正后 **YOLOv8m 0.8886** 胜出（YOLO11m 0.8848、YOLOv10m 0.7313 淘汰），同时证明 `thr_095×3 + thr_080×1` 配方在 YOLO 上依然有效（0.8766 → 0.8886，+1.2 个点）。
