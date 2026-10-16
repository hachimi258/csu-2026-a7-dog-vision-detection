# B 榜 Step7：HSV 规则标注 → 自举迭代 + 人工难例修正

> **归档说明**：本文由项目全程技术复盘按"成果主线"重组而成，对应原复盘文档的相关章节（源文件 `archive/26中国软件杯_项目全程复盘.md`）。
> **性质**：基于代码、日志、产物实测的复盘记录，**非该阶段原始源码**；文中的代码片段为**原样节选（excerpt）**，用于说明关键实现，可能省略了与本节无关的部分。

## 目的

Step6 证明"伪标签占 98.7% + 单次推理"的数据质量太差。Step7 换一条路线：

> **从高置信度纯净种子出发，做多轮"训练 → 推理 → 合并"的自举迭代（bootstrapping），并在收敛后人工修正难例。**

## ⚠️ 被遗忘的前奏：`hsv.py` 规则标注的失败

`step7/hsv.py`（8451 B，日志 `20260630_143337_hsv.log`）是 Step7 的**第一个尝试**，比自举迭代早两天。

**思路**：完全放弃学习式标注器，用**手工调优的 HSV 颜色阈值**直接生成 YOLO 标签。

**火焰颜色规则（精确）**：

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([5, 80, 120])      # H=5,  S=80,  V=120
upper = np.array([30, 255, 255])    # H=30, S=255, V=255
mask = cv2.inRange(hsv, lower, upper)
```

在 OpenCV 的 HSV 约定里 **H ∈ [0,180]**（度数/2），所以 **H = 5~30 对应 10°~60°** —— 橙到黄，即火焰核心/主体而非红色尖端（红是 H < 5 或 > 175）。饱和度下限 80 与亮度下限 120 排斥发白和暗的区域。

**形态学与区域选择**：

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)     # 闭运算桥接闪烁造成的断裂
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_contour = max(contours, key=cv2.contourArea)           # ⭐ 只取最大连通域
```

**8 像素外扩**（精确，注意 `bw`/`bh` 是对**已平移的** `x`/`y` 做钳制，右/下不会越界）：

```python
pad = 8
x  = max(0, x - pad)
y  = max(0, y - pad)
bw = min(w - x, bw + 2*pad)
bh = min(h - y, bh + 2*pad)
```

**运行结果（`20260630_143337_hsv.log`）**：

```
训练集sample图片数: 70
验证集sample图片数: 30
captures 图片总数: 2827
captures 有火图片数: 2827 (跳过 0 张)      ← 2827/2827 = 100% 都"有火"！
sample训练集有火: 70/70
sample验证集有火: 30/30
train.txt 写入 2897 条
val.txt 写入 30 条
总图片数（有火）: 2927
请人工检查 images/ 和 labels/ 中的标注，删除误检图片和对应标签（预计 <20 张）
```

**❌ 这个方案失败了**：

- **2827 张全部被判定为"有火"**（0 张跳过）—— 因为 HSV 规则**没有最小面积过滤**，一个 5×5 闭运算后的孤立橙色像素就能生成合法标签
- **2927 张全部是正样本，零负样本** —— 一个只有正类的数据集
- 输出 2897 训练 + 30 验证，**验证集只有 30 张**（且只包含 HSV 找到火的那些，因为无火的图根本不复制）
- 脚本最后一行是**人工 QC 指令**："请删除误检图片和对应标签（预计 <20 张）" —— 作者预期少量误检，实际是 100% 误检。**这个人工修剪从未自动化，也从未执行**
- 另外：`process_image(src, dst_img, dst_label, is_train)` 的 `is_train` **参数从未被使用**（死参数）；脚本**没有 `if __name__ == "__main__":` 保护**，import 就会跑完 2827 张；`SAMPLE100_DIR` 定义后从未使用

**结局**：`step7/hsv.py` 生成的 `data1` 在 2026-06-30 生成，随后在 **2026-07-02 05:07 / 05:48 / 05:55** 被 `init.py` **三次覆盖**（三个 `init.log`，最终版本 train=892/val=191/test=192）。HSV 方案被彻底放弃。

> **这个失败值得记录**：它是"规则方法在真实场景中难以标定"的教科书案例 —— 火焰的颜色与**白炽灯、木色、日落、皮肤**高度重叠，在没有面积/形状先验和负样本约束时，规则会把整个数据集标成正样本。

## 起点与种子构建（`tool/init.py`）

**起点权重**：`step5/best1/yolov8m_AdamW_1e-4/train/weights/best.pt`（Step4 最优，mAP50=0.9888）

**种子来源**：`step2/final_data/annotations/pseudo/thr_095/*.xml` 中 `confidence >= 0.99` 的标注

| 参数 | 值 |
|:---|:---|
| `CONF_THR` | **0.99** |
| `SEED` | **42** |
| 图片池 | `captures/` 2827 张 |
| 划分 | 随机打乱，7 : 1.5 : 1.5（`int()` 截断） |
| 输出 | `data1/`（**物理复制**全部 2827 张图） |
| 标签格式 | 硬编码 `img_w=640, img_h=480` → `0 cx cy w h`（6 位小数） |

**`init.py` 结果**（`20260702_055547_init.log`）：

```
Total XMLs: 1275
Valid seeds (conf>=0.99): 1275      ← 注意：1275 全部"通过"
Skipped - no image: 0
Skipped - no bbox: 0
Skipped - low confidence: 0
Total images copied: 2827
Train: 892, Val: 191, Test: 192
```

## ⚠️ 勘误："≥0.99 精选种子"是假的

`S7conclusion.md` 与 `源码清单与说明.md` 都称"筛选 ≥0.99 得到 **1275** 个纯净种子"。但：

- `thr_095` 目录本身就只有 **1275** 个 XML
- `Valid seeds = 1275` 且 `Skipped - low confidence: 0` ⇒ **筛选没有淘汰任何一个**

**根因**（`step7/tool/init.py` 第 57 行）：

```python
conf_el = obj.find("confidence")
conf = float(conf_el.text.strip()) if conf_el is not None and conf_el.text else 1.0
```

而 `step2/gen_labels.py` 的 `create_xml()` **根本没有写入 `<confidence>` 字段**（它只写 `name/pose/truncated/difficult/bndbox`）。所以**每个 object 的 conf 都默认成 1.0**，全部 ≥0.99。

> **真实含义**：Step7 的种子集实际是 **`thr_095` 的全部 1275 张**，不是"≥0.99 的精选子集"。**整个 Step7 的叙事基础（"从纯净高置信度种子出发"）与代码不符。**
>
> **侧面印证**：Step8 的 `init.py` 用 `conf >= 0.9` 从**模型重新推理**得到 2079 框 —— 说明直接从模型拿置信度才是可行做法。Step7 试图从 Step2 的 XML 拿置信度，但那个字段从来不存在。

## 参数：训练配置（`tool/train.py`，硬编码）

```python
HP = {
    "imgsz": 640, "epochs": 20, "batch": 16,
    "optimizer": "AdamW", "lr0": 1e-4,
    "momentum": 0.9, "weight_decay": 0.0005,
    "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
    "patience": 0,          # 关闭早停
}
# device=0, amp=False（避免 AMP check 下载 yolo26n.pt 卡住）
# OOM 时自动降 batch 到 8 重试
```

**数据增强全部关闭**（`mosaic=0, mixup=0, copy_paste=0`）—— 因为伪标签质量不高，强增强会放大噪声。

**关键工程细节**：训练前 `os.chdir(data_path)`，让 YOLO 正确解析 `dataset.yaml` 中的**相对路径**（`path: .`, `train: train`, `val: val`）。

> **踩坑记录**：原先用 `train.txt` 列表式配置（`train: train.txt`）时，Windows 的 `os.sep`（`\`）与 YOLO 期望的路径分隔符不匹配，导致 **892 张图全部识别为 background**（`cls_loss=237.6`）。改用**目录格式**后，YOLO 内部的 `\images\` → `\labels\` 字符串替换机制正常工作。**这是 Windows + Ultralytics 的经典坑。**

## 实现：自举迭代两轮

**迭代流程**：

```
data{n} 训练 → model_v{n} → 对 2827 张全量推理 → pseudo_round{n}/（分 5 档阈值）
    → merge 高置信档（thr_098）进种子 → data{n+1} → 下一轮
```

**`gen.py`**（伪标签生成，分层写出）：

```python
TIERS = [("thr_099",0.99), ("thr_098",0.98), ("thr_095",0.95),
         ("thr_090",0.90), ("thr_000",0.10)]
# 推理参数：conf=0.1, iou=0.5, imgsz=640
# 每图取面积最大的框 → 归一化 (cx,cy,w,h)，**硬编码**除以 640/480
# 同时写 metadata.json：{stem: max_conf}
```

**`merge.py`**（合并规则）：

```python
split_seed = round_num * 10 + 42        # 每轮划分不同（R1→52, R2→62）
for stem, content in new_labels.items():
    if stem in merged:
        seed_c = 1.0                      # 种子置信度视为 1.0
        new_c  = float(new_conf.get(stem, 0.0))
        if new_c > seed_c:                # ⚠️ 死分支：new_c 恒 ≤ 1.0，永不成立
            merged[stem] = content
        conflicts += 1
```

> **⚠️ 死分支**：`new_c > 1.0` 永远为假（`gen.py` 的置信度上限是 1.0），所以**新标签永远无法覆盖种子**。这个 `if` 在功能上等价于注释。作者自己在 `S7计划2.md` 的陷阱清单里写了"新标签不可能超过 1.0，所以种子永远优先"—— 但代码里仍保留了这个不可能成立的分支。

**`converge.py`**（收敛判定，**图片数口径**）：

```python
prev_set / curr_set = {stem of *.txt}
new_added   = len(curr_set - prev_set)
growth_rate = new_added / len(prev_set) * 100
converged   = (new_added < 30) or (growth_rate < 2.0)     # threshold 默认 30（图片数！）
```

**⚠️ 与 Step8 的收敛定义完全不同**：Step7 是**图片数**口径（`<30` 张或 `<2%` 增长），Step8 是**框数 + IoU 匹配**口径（`≤5%`）。原文档只描述了 Step8 那一个。

**迭代结果**：

| 轮次 | 数据集 | 标签数 | 训练样本 | 训练耗时 | mAP50 | 检出图数 | 平均置信度 | thr_098 新增 |
|:---|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|
| 起点 | — | — | — | — | **0.9888** | — | — | — |
| **R1** | data1 | 1275 | 892 | 4.1 h | **0.9882** | 2120 | 0.8335 | **82** |
| **R2** | data2 | 1276 | 893 | 4.2 h | **0.9913** | 2138 | 0.8608 | **0** ✅ |

R1/R2 的 `gen.py` 分层统计：

```
R1: thr_099: 0   thr_098: 82   thr_095: 667   thr_090: 1409   thr_000: 2120
R2: thr_099: 0   thr_098: 0    thr_095: 716   thr_090: 1529   thr_000: 2138
```

`merge.py` R1 日志：

```
Seed dir: data1 (1275 labels)
New dir:  pseudo_round1/thr_098 (82 labels)
Conflicts: 81            ← 82 个新标签中 81 个与种子同名（被种子覆盖）
Merged total: 1276       ← 只净增 1 张！
Split seed: 52
Train: 893, Val: 191, Test: 192
```

> **⚠️ 收敛判定的实质**：R1 的 82 个 `thr_098` 标签中有 81 个与种子冲突（种子胜出），**净增只有 1 张图**。原文档说"迭代扩展"，实际上**从 R1 开始数据集就基本饱和**。R2 的 `thr_098 = 0` 说明模型已不再产生任何 ≥0.98 的新检测。
>
> **真实解读**：这不是"自举逐步提升"，而是**模型已经收敛到与种子几乎一致的输出**。`converge.py` 输出 `CONVERGED` 是合理的，但原文档"自举已生效并自然收敛"的因果叙述缺少"净增 1 张"这个关键事实。

## 人工难例修正

对 `pseudo_round2/thr_000`（2138 张全量检出）人工审查，用 **LabelMe** 标注了 **9 张难例**（存于 `step7/pseudo_round2/fix/`）：

| 图片 | 问题 |
|:---|:---|
| `frame_000482`, `frame_000483` | 伪标签框位置不准确 |
| `frame_001755`, `frame_001990`, `frame_001991`, `frame_001992` | 伪标签框偏差较大 |
| `frame_002159`, `frame_002160`, `frame_002161` | 伪标签框需修正 |

## 实现：全量数据整合（`tool/build4.py`）

```
data3/labels (1276 seed)
  + pseudo_round2/thr_000 (2138)
  + pseudo_round2/fix/*.json (9 个人工修正) ── 直接覆盖 thr_000 中同名标签
  → 合并（seed 优先）
  → data4 = 2139 unique labels
```

`build4.py` 的 `json_to_yolo()`（**⚠️ 只取 `shapes[0]`**）：

```python
s = shapes[0]                       # 只处理第一个 shape —— 多框会被静默丢弃
x1,y1 = pts[0]; x2,y2 = pts[1]
xmin, xmax = min(x1,x2), max(x1,x2) # 处理 LabelMe 坐标方向可能反的问题
ymin, ymax = min(y1,y2), max(y1,y2)
cx = (xmin+xmax)/2/640; cy = (ymin+ymax)/2/480
w  = (xmax-xmin)/640;   h  = (ymax-ymin)/480
return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
```

**合并顺序**：`dict(seed_labels)` → fixes（覆盖 thr_000 中的同名）→ R2 标签（seed 冲突时胜出）。**seed 赢下所有冲突。**

`20260703_002547_build4.log`：

```
Fix labels: 9
Seed labels (data3): 1276
Pseudo R2 labels (thr_000): 2138
Merged total: 2139 (conflicts: 1275)     ← 2138 中 1275 与种子冲突
Train: 1497, Val: 320, Test: 322
```

**磁盘实测验证**：`data4/labels` = **2139** 个文件，`train/images` = **1497**，`val/images` = **320**，`train.txt`/`val.txt`/`test.txt` = 1497/320/322 行 ✅

**⚠️ `json_to_yolo` 只取 `shapes[0]` 的风险**：如果人工用 LabelMe 标了多个框（一图两处火），**第二个之后的框会被静默丢弃**。9 张修正样本恰好都是单框，所以没暴露。

## 最终训练与提交包

| 项 | 值 |
|:---|:---|
| 数据 | `data4`（2139 标签，1497 / 320 / 322） |
| 权重起点 | `step5/best1/yolov8m_AdamW_1e-4/best.pt`（**回到 Step4 权重，不是 model_v2**） |
| epochs | 20 |
| batch | 16 |
| 耗时 | ~6.5 h |
| **验证 mAP50** | **0.9877** |
| 权重 | `step7/result/model_v4/train/weights/best.pt`（49.6 MB） |

> **注意**：最终训练**从 Step4 权重重新开始**（而不是用 R2 的 `model_v2`），只在 `data4` 上跑 20 epoch。所以"自举迭代"的产物（`model_v1`、`model_v2`）**没有进入最终模型** —— 它们只用于生成伪标签。

**ONNX 导出**（`20260703_050502_export.log`）：

```
weights: .../model_v4/train/weights/best.pt
onnx:    .../model_v4/train/weights/best.onnx
size:    103625064 bytes (98.8 MB)     ← opset=11, imgsz=640, 输出 [1,5,8400]
```

`step7/tool/pack.py` 写入的 `infer_cfg.yml`（手写模板）：

```yaml
mode: fluid
draw_threshold: 0.5
metric: COCO
arch: YOLO
min_subgraph_size: 3
NormalizeImage: {is_scale: false, mean: [0,0,0], std: [1,1,1]}
label_list: [firebig]
use_dynamic_shape: false
target_size: 640
```

**提交包**：`step7/upload/20260703_050648/submission.zip`（84,282.7 KB = 82.31 MB），内含：

| 文件 | 大小 |
|:---|:---|
| `predict.py` | 5.5 KB（IMGSZ=640，去 NMS，GPU 优先 + CPU 回退） |
| `model/model.pdmodel` | 164.3 KB |
| `model/model.pdiparams` | **101,107.5 KB = 98.74 MB** |
| `model/infer_cfg.yml` | 0.3 KB |

## ⚠️ Step7 的未验证结局

Step7 产出就绪、`mAP50=0.9877`，但 `S7conclusion.md` 的"**预估平台 FPS ~35-45**"**从未经过任何实测**：

- `step7/log/` 里**没有任何平台 FPS 记录**
- `step7/upload/20260703_050648/` 里只有本地测试残留（`test_10.txt` 0.2 KB / `result_test.json` 0 字节），说明只跑过 10 张的本地试探
- `上传/` 下没有平台反馈文件

而 `源码清单与说明.md` 与 `提交总说明.md` 后来给出的结论是：

> "Step7及之后版本尝试提升精度但导致 FPS 不合格（IMGSZ=640 使推理过慢）"

**⚠️ 这个结论在项目内找不到任何日志或平台反馈支撑。**

**从物理上推算**：Step5（640 + NMS）平台 14.35 FPS。Step6 的 P1 vs 基线的本地 FPS 是 7.10 vs 约 7.4（**NMS 几乎不耗时**），说明真正的提速来自 512 分辨率。所以 **640 + 去 NMS 的平台 FPS 很可能仍在 14~18 之间，确实危险** —— 但这是推断，不是实测。

> **如果当时做对了**：Step7 完成转换后**立即提交一次平台**（成本只是一次压缩上传），就能拿到真实的 640 FPS，避免在"精度还是速度"之间盲猜。**这一步的缺失直接导致了 Step8/Step9 共约 40 小时 GPU 时间的方向性不确定。**

## 结论

Step7 先以 `hsv.py` 规则标注尝试直接生成标签，结果 2827 张 100% 被判"有火"、零负样本而彻底失败（`data1` 后被 `init.py` 三次覆盖）；随后改为两轮自举迭代，但 R1 的 82 个新标签中 81 个与种子冲突、**净增仅 1 张**，R2 的 `thr_098` 为 0 即已饱和。经 9 张 LabelMe 人工难例修正后固化出 `data4`（2139 标签，1497/320/322），从 Step4 权重重训 20 epoch 得验证 mAP50 0.9877 并打包 82.31 MB；但**该包从未提交平台，FPS 35-45 与"精度提升导致 FPS 不合格"的结论均无任何实测支撑**。
