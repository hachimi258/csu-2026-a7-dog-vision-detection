# B 榜 Step8：多框感知三阶段自举（15 轮）

> **归档说明**：本文由项目全程技术复盘按"成果主线"重组而成，对应原复盘文档的相关章节（源文件 `archive/26中国软件杯_项目全程复盘.md`）。
> **性质**：基于代码、日志、产物实测的复盘记录，**非该阶段原始源码**；文中的代码片段为**原样节选（excerpt）**，用于说明关键实现，可能省略了与本节无关的部分。

## 目的

Step7 的模型是**单框模型**（每图只学一个最大框）。Step8 尝试把它改造成**多框感知模型**，并产出 LabelMe 格式的全火焰框数据集，用于人工核验与后续研究。

## 起点（`tool/init.py`）

| 项 | 值 |
|:---|:---|
| 起点权重 | `step7/result/model_v4/train/weights/best.pt` |
| 做法 | **用模型对 2827 张重新推理**（不再依赖 Step2 的 XML —— 修正了 Step7 的错误） |
| 置信度阈值 | **0.90** |
| 推理参数 | `conf=0.1, iou=0.5, imgsz=640, verbose=False` |
| 归一化 | **硬编码 `/640.0`, `/480.0`** |
| 划分 | **时序均匀**（见下） |
| 输出 | `data1/`（`copy2` 复制全部 2827 张图） |

**时序划分算法**（`temporal_split_indices`）：

```python
step = max(1, int(n_total / (n_total * VAL_RATIO)))   # VAL_RATIO=0.15 → int(6.667) = 6
val_idx  = set(range(0, n_total, step))               # 0, 6, 12, ...
test_idx = set(range(step // 2, n_total, step))       # 3, 9, 15, ...
train_idx = sorted(set(range(n_total)) - val_idx - test_idx)
```

n = 2827 → **train 1884 / val 472 / test 471**

`20260703_105750_init.log`：

```
Total images: 2827
Images with boxes (conf>=0.9): 2075
Total boxes: 2079
Train: 1884 imgs (1382 labeled)
Val:   472 imgs (346 labeled)
Test:  471 imgs (347 labeled)
```

> **注意**：`data1/images` 包含**全部 2827 张图**（含 752 张无标注的），而 train/val/test 目录只含被划分的 2827 张中的对应子集，其中未标注的图**没有标签文件**。
>
> `data1/metadata.json` schema：`{stem: {ext, box_count, max_conf, boxes:[{cx,cy,w,h,conf,x1,y1,x2,y2}]}}`
>
> `TRAIN_RATIO`/`TEST_RATIO` 常量被定义但**未使用**。

## 实现：三阶段 15 轮迭代

### Phase 1（R1–R4）：单框 → 多框发现

| Round | 数据集 | 训练 mAP50 | 收敛率 | 检出框 | 均框/图 | 关键事件 |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| R1 | data1 | 0.9931 | — | 2206 | 1.02 | 种子（2075 图 / 2079 框） |
| R2 | data2 | 0.9949 | **20.99%** | 2430 | 1.03 | 首批多框（80 张 2 框） |
| R3 | data3 | 0.9943 | 5.24% | 2572 | 1.06 | 首次 3 框（1 张） |
| R4 | data4 | 0.9945 | **4.37%** ✅ | 2568 | 1.05 | 收敛 |

**R2 收敛日志**（`20260703_151331_converge.log`）：

```
Previous (pseudo_round1/thr_090): 1853 images, 1868 boxes
Current  (pseudo_round2/thr_090): 2200 images, 2252 boxes
New boxes (IoU<0.5 vs prev): 392
Rate: 392/1868 = 0.2099 (20.99%)   Threshold: 5.0%   Verdict: NOT_CONVERGED
```

**框数分布演变**：

| Round | 1 框 | 2 框 | 3+ 框 | min/max |
|:---|:---:|:---:|:---:|:---|
| R1 | 2136 | 35 | 0 | 1/2 |
| R2 | 2270 | 80 | 0 | 1/2 |
| R3 | 2289 | 140 | 1 | 1/3 |
| R4 | 2346 | 111 | 0 | 1/2 |

**产物**（`tool/build.py`，`20260703_160359_build.log`）：

```
Input: pseudo_round4 + data1
Output: step8/labelme_dataset
Total images: 2827
Images with boxes: 2457
Total boxes: 2508
Avg boxes per labeled image: 1.02
Images without boxes: 370
```

> **关键发现**：单框模型**天然抑制多框输出**。即使用多框标签训练，多框能力增长极慢——收敛时仅 111 张图有 2+ 框。
>
> **`build.py` 的两个问题**（原文档未记录）：
> 1. `last_round = sorted(STEP8.glob("pseudo_round*"))[-1]` —— **字典序排序**！`pseudo_round9` 会排在 `pseudo_round15` 之后。在 ≥10 轮时"最后一轮"会选错
> 2. **没有任何置信度过滤**，尽管文档 T9 声称构建 labelme_dataset 时加了 `conf≥0.3` 二次过滤（代码里不存在）
> 3. 为**全部 2827 张图**都写 LabelMe JSON —— 无框的 370 张写 `"shapes": []`

**`build.py` 的 view 配色**（⚠️ 与文档不符）：

```python
if   conf >= 0.95: color = (0, 0, 255)      # 红
elif conf >= 0.90: color = (0, 165, 255)    # 橙
elif conf >= 0.80: color = (0, 255, 255)    # 黄
else:              color = (255, 0, 0)      # 蓝
```

> `S8conclusion.md` 声称配色是 **🔴≥0.95 🟠≥0.8 🟡≥0.5 🟢<0.5** —— 实际代码阈值是 **0.95 / 0.90 / 0.80**。**勘误。**

### Phase 2（R5–R12）：多框强化

**转折点**：人工用 LabelMe 标注了 **339 张多框图片**（保存在 `step8/labelme_dataset5/multi1/`，磁盘实测 678 个文件 = 339 jpg + 339 json）。

**核心策略调整**：

1. **多框图片过采样**：多框图物理复制 ×2（文件名加 `_r2` 后缀）
2. **单框图片欠采样**：随机取 50%
3. 零框图片全部保留（提供负样本）
4. 时序均匀划分验证集（472 张）
5. `step8/tool/train.py` 额外支持 `--iou` / `--batch` / `--epochs` 参数

| Round | 训练 mAP50 | 收敛率 | 检出框 | 均框 | 2+框数 | 关键事件 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| R5 | 0.9428 | **78.41%** | 4841 | 1.81 | 1456 | 多框"爆炸" |
| R6 | 0.9679 | 19.96% | 5417 | 1.97 | 1704 | 持续增长 |
| R7 | 0.9548 | 8.84% | 4970 | 1.86 | 1469 | 趋稳 |
| R8 | 0.9557 | 4.36% ✅ | 4777 | 1.79 | 1421 | （判定收敛但继续） |
| R9 | **0.9706** | 37.29% | **6060** | **2.17** | 1883 | **首次 7 框！** |
| R10 | **无训练日志** | 13.79% → 10.43% | 6385 → 5935 | 2.27 → 2.11 | 2029 → 1895 | 两次 gen，收敛未过 |
| R11 | 0.9711 | 5.69% | 5487 | 1.94 | 1696 | — |
| R12 | **0.9794** | **3.70%** ✅ | 4966 | 1.76 | 1355 | **Phase 2 收敛** |

**R5 的爆发**（`20260703_180511_gen.log`）：

```
Total boxes detected: 4841        ← R4 是 2568，翻了近一倍
Avg boxes per image: 1.81
Box count distribution: min=1, max=5
  1box=1223, 2box=974, 3+box=482
```

对比 R4：`1box=2346, 2box=111, 3+box=0`。**多框检测能力被一次性激活**。

**R9 的峰值**（`20260704_112229_gen.log`）：

```
Total boxes: 6060
Avg boxes/image: 2.17
Box dist: {1: 906, 2: 979, 3: 562, 4: 223, 5: 97, 6: 21, 7: 1}
```

> **⚠️ 严重缺口**：`step8/log/` 中**没有 R9/R10/R13 的训练日志**。`20260704_112229_gen.log`（R9）之后直接跳到 `20260704_161250_train.log`（记 R11）。同时 R9→R10 和 R10→R11 的 `converge.py` 时间戳都**早于**后续训练日志，说明**收敛判定与训练存在时序错位**。
> 因此 R9/R10/R13 的 mAP50 无日志可查 —— 原文档表格中 R10 那行缺失 mAP50 是诚实的。

### Phase 3（R13–R15）：IoU=0.7 召回验证

**目的**（原文档）：验证降低 IoU 阈值能否找回更多漏检框。

| Round | 训练 mAP50 | 收敛率 | 检出框 | 均框 | 结论 |
|:---|:---:|:---:|:---:|:---:|:---|
| R13 | **无训练日志** | 5.89% | 4987 | 1.77 | IoU 切换后行为变化 |
| R14 | 0.9803 | 6.87% | 5008 | 1.78 | 收敛中 |
| R15 | **0.9833** | **4.94%** ✅ | 4844 | 1.73 | 收敛 |

**Phase 2 vs Phase 3 对比**（原文档结论）：

| 维度 | Phase 2 (R12) | Phase 3 (R15) | 优胜 |
|:---|:---|:---|:---|
| 模型裸检出（conf≥0.3，500 张采样） | **994 boxes** | 953 boxes | Phase 2 |
| 均框/图 | **2.02** | 1.94 | Phase 2 |
| 多框（2+）图片 | **303** | 282 | Phase 2 |
| 收敛率 | **3.70%** | 4.94% | Phase 2 |
| 训练收敛轮数 | 8 轮 | 3 轮 | Phase 3 |

**最终选择**：Phase 2 的 `model_v12`。

> **但见下文"`--iou 0.9` 从未生效"**：Phase 2 / Phase 3 的区分条件（`--iou`）**在代码层面从未生效**，所以这个"对比"在方法学上不成立。

## 数据集演进

| 数据集 | 图片数 | 总框数 | 均框 | 多框(2+) | 来源 |
|:---|:---:|:---:|:---:|:---:|:---|
| `labelme_dataset` | 2827 | 2508 | 0.89 | 111 | Phase 1 收敛 |
| `labelme_dataset5` | 2827 | 3057 | 1.08 | 339 | 人工标注种子（+`multi1/`） |
| `labelme_dataset8` | 2827 | 3521 | 1.25 | 473 | Phase 2 中期 |
| **`labelme_dataset12`** | **2825** | **3713** | **1.31** | **510** | **Phase 2 收敛（最优）** |
| `labelme_dataset15` | 2826 | 3714 | 1.31 | 510 | Phase 3 收敛 |

**⚠️ `labelme_dataset4` 不存在**（原文档 `S8conclusion.md` §2.1 与 §五 都引用了它）。磁盘实测存在的是：`labelme_dataset`、`labelme_dataset5`、`labelme_dataset8`、`labelme_dataset12`、`labelme_dataset15`。

**各 labelme_dataset 的真实目录结构（磁盘实测）**：

| 数据集 | 结构 |
|:---|:---|
| `labelme_dataset` | `images/`（2827 jpg + 2827 json **同目录**）、`annotations/`（2827 json）、`view/`（2827） |
| `labelme_dataset5` | `images/`（2827 jpg + 2492 json）、`labels/`（2492 txt）、`train/`、`val/`、`multi1/`（339+339）、`code/` |
| `labelme_dataset8` | `view/`（2685）—— **扁平目录 + view** |
| **`labelme_dataset12`** | **根目录下 2825 jpg + 2825 json（同目录）**，`view/`（2825） |
| `labelme_dataset15` | 根目录下 2826 jpg + 2826 json，`view/`（2826） |

> **⚠️ 勘误**：`S8conclusion.md` §2.1 声称 `labelme_dataset12/` 含 `images/` + `annotations/` 子目录 —— **那是基础版 `labelme_dataset/` 的结构**。`labelme_dataset12` 是图片与 JSON 同目录（LabelMe 的默认习惯）。

**LabelMe JSON 格式**（实测 `labelme_dataset12/frame_000482.json`，1.3 KB）：

```json
{
  "version": "5.0.1",
  "flags": {},
  "shapes": [
    {"label": "fire", "points": [[246.51, 64.78], [426.90, 249.85]],
     "group_id": null, "shape_type": "rectangle", "flags": {}},
    {"label": "fire", "points": [[397.48, 88.28], [527.25, 221.49]],
     "group_id": null, "shape_type": "rectangle", "flags": {}}
  ],
  "imagePath": "frame_000482.jpg",
  "imageData": null,
  "imageHeight": 480,
  "imageWidth": 640
}
```

## ⚠️⚠️ 过采样策略：脚本已丢失，策略只能从产物反推

**重要发现**：**整个仓库里不存在任何过采样代码。**

对所有 `*.py`（排除 `PaddleDetection`、`result_sample`、`site-packages`）grep `_r2` / `_r{` / `suffix` / `defaultdict` / `random.sample` —— **零匹配**。没有 `build*.py` / `*sample*.py`，没有任何脚本引用 `data12` / `labelme_dataset12`，`data12`~`data15` **没有 `code/` 目录**（对比 `data1`~`data6` 都有）。

**⇒ 生成 `data12` 的那个脚本没有被保留。只有数据和文档幸存。**

**⚠️ 同时**：`S8conclusion.md` §4.1 的脚本表里把过采样逻辑归给了 `tool/rebuild.py`（"含 oversampling 逻辑"）—— 但 `rebuild.py` 的 212 行里**完全没有过采样代码**，它只做 JSON→YOLO 转换和划分。**勘误。**

**策略可以从产物精确复原并对上账**：

**① 文件名留下了指纹**（`data10/train/images` 抽样）：

```
frame_000000.jpg
frame_000000_r2.jpg      ← 多框图第 1 次复制
frame_000000_r10.jpg     ← 单框图随机采样命中
frame_000000_r11.jpg
frame_000000_r12.jpg
```

**② `data12` 的算术完全对得上**：

| 项 | 数值 | 推导 |
|:---|:---|:---|
| val | **472** | 精确等于 `sorted(all_stems)[0::6]`；全部有标注；**无 `_r2`** |
| train images | **2179** | = 1669 base + **510 `_r2`** |
| `_r2` 数 | **510** | 精确等于文档记载的 510 张多框图 ⇒ **多框 ×2 确认** |
| 不在 train/images 的 label 数 | **1158** | ≈ 2315 单框图 / 2 = 1157.5 ⇒ **单框 50% 欠采样确认** |
| train labels | **2177** | = 510 + 510 + 1157 |
| train 中仅有的 2 张无标签图 | `frame_000491`, `frame_001103` | 零框负样本，全部保留 |

**各数据集 train 规模对比**：

| 数据集 | train jpg | 其中 `_r*` | train labels | 全部 labels | val jpg |
|:---|:---:|:---:|:---:|:---:|:---:|
| data1 | 1884 | 0 | — | 2075 | 472 |
| data6 | 2355 | 0 | 2082 | 2492 | 472 |
| data9 | 2194 | **473** | 2052 | 2685 | 472 |
| **data10** | **7240** ⚠️ | 5555 | 7202 | 2789 | 472 |
| data11 | 2180 | 510 | 2176 | 2823 | 472 |
| **data12** | **2179** | **510** | 2177 | 2825 | 472 |
| data13/14/15 | 2179 | 510 | 2177/2178 | 2825/2826 | 472 |

**③ ⚠️ `data10` 是命名冲突 bug 的现场**：

```
data10/train/images = 7240 = 1685 base + 505 copies × ( _r2 … _r12 )  共 11 种后缀
                                                          ↑ max suffix = 12
```

**同一批 505 张多框图被重复复制了 11 次** —— 这正是 `S8conclusion.md` 陷阱 T7 描述的"oversampling 命名冲突：`_r` 后缀的文件被反复追加"。**而 T7 声称的 `defaultdict(int)` 唯一命名修复在任何脚本里都不存在** —— 因为那个脚本本身就没被保留，修复也就无从谈起。

**④ ⚠️⚠️ 验证集泄漏**：

| 项 | 数值 |
|:---|:---|
| `data12/train/labels` 去掉 `_r` 后的唯一名字 | 1667 |
| `data12/val/labels` | 472 |
| **`train ∩ val`** | **264** |
| val 中在 train 里也有 `_r2` 孪生图的 | **96** |

**原因**：train 是从过采样/欠采样规则**组装**出来的，而不是作为 val 的**补集**算出来的。`S8conclusion.md` §4.4 声称"`train_idx = 剩余`"—— **交付的数据不是这样**。

对照：`step7/data4` **没有**这种重叠（它的划分是一次 shuffle 后的不相交切片）。

> **这直接解释了为什么 Step8 的验证 mAP50 高达 0.94~0.98 却对实用没有对应的收益。**

**⑤ ⚠️ 950 个标签未参与任何 split**：`data12/labels` 有 2825 个文件，train 2177 + val 472 = 2649，差 176（算上 `_r` 别名则更多）。**部分标注数据实际未被使用。**

## ⚠️⚠️⚠️ 勘误：`--iou 0.9` 从未生效，且概念被误解

原文档（`S8conclusion.md` L84、`项目详细技术方案.md` L217）反复宣称：

> "**IoU=0.9 严格正样本匹配**，只学习高质量锚框 → 反而提升多框检测精度"

**代码证据**（`step8/tool/train.py`）：

```python
parser.add_argument("--iou", type=float, default=0.0)
...
iou_val = args.iou if args.iou > 0 else 0.7
log(f"Batch: {batch}, Epochs: {epochs}, IoU: {iou_val}")
try:
    results = model.train(
        data=..., imgsz=640, epochs=epochs, batch=batch,
        optimizer='AdamW', lr0=1e-4, momentum=0.9, weight_decay=0.0005,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, patience=0,
        project=..., name='train', exist_ok=True,
        device=0, amp=False,
        iou=iou_val,                                  # ← 传给 Ultralytics
    )
except RuntimeError as e:
    if "out of memory" in str(e).lower() or "OOM" in str(e):
        log(f"OOM with batch={batch}, retrying with batch=8")
        batch = 8
        results = model.train(
            data=..., imgsz=640, epochs=HP["epochs"], batch=batch,
            ... ,
            device=0, amp=False,
            # ⚠️⚠️ 这里没有 iou=iou_val！
        )
```

**两个独立的错误**：

**错误 A：概念误解。** Ultralytics 的 `iou` 参数是 **NMS 的 IoU 阈值**（用于 `non_max_suppression`），**不是训练时的正样本匹配阈值**。后者在 YOLOv8 里是 TAL 的 `topk`/`alpha`/`beta`（或 v5 时代的 `anchor_t`）。

**所以 `--iou 0.9` 的真实效果**：

1. **训练期**：`iou` 写入 `args`，主要影响**训练过程中每个 epoch 的验证 NMS**。把 NMS IoU 从 0.7 提到 0.9 意味着**保留更多重叠框** ⇒ 验证 mAP 会**偏高**（宽松去重）
2. **推理期（`gen.py`）**：`gen.py` 里**硬编码了 `iou=0.5`**，与训练时的 `iou` 无关
3. **多框能力的真正来源**：是 **R5 前引入的 339 张人工多框标注 + 多框图 ×2 过采样**，不是 `--iou 0.9`

**旁证**：Phase 2 的验证 mAP50 在 0.94~0.98 之间（比 Phase 1 的 0.993~0.995 **更低**），而 Phase 3（文档说 IoU=0.7）又回到 0.980~0.983。如果 `iou=0.9` 真的提升精度，Phase 1（0.7）应该更低才对。**mAP50 的波动更可能只是过采样改变了训练分布。**

**错误 B：OOM 分支漏传参数 ⇒ 实际上 `--iou 0.9` 从未生效。**

**日志证据**：`step8/log/` 里**所有轮次**的 train 日志都写着 `Batch used: 8`（不是默认的 16）：

```
20260703_144603_train.log: Batch used: 8    ← R1
20260703_151057_train.log: Batch used: 8    ← R2
...（全部 12 个 train 日志都是 8）
20260704_215212_train.log: Batch used: 8    ← R15
```

**`Batch used: 8` 意味着每一轮都走了 OOM 重试分支** ⇒ **每一轮的 `iou` 都静默回退到 0.7**。

> **⚠️⚠️⚠️ 最终结论：Step8 全部 15 轮的 `--iou 0.9` 从未生效过。**
>
> 原文档中"Phase 2 用 IoU=0.9，Phase 3 切回 IoU=0.7"的对比实验，**在代码层面等价于同一个配置**。这个对比实验在方法学上**不成立**。
>
> 另外 OOM 分支还把 `epochs` 重置成了 `HP["epochs"]`（20），忽略 `--epochs` 参数。

## 实现：收敛判定机制（Step8，与 Step7 不同）

`step8/tool/converge.py` 的**框级 IoU 匹配口径**：

```python
def count_new_boxes(prev_meta, curr_meta, iou_thr=0.5):
    """统计 curr 中未能与 prev 任何框 IoU≥0.5 匹配的框数"""
    for stem, curr_data in curr_meta.items():
        curr_boxes = [(b["x1"], b["y1"], b["x2"], b["y2"]) for b in curr_data["boxes"]]
        if stem not in prev_meta:
            new_count += len(curr_boxes)      # 新图贡献全部框
            continue
        prev_boxes = [...]
        for cb in curr_boxes:
            if not any(iou(cb, pb) >= iou_thr for pb in prev_boxes):
                new_count += 1

# 只统计 max_conf >= 0.90 的图
prev_090 = {s: d for s, d in prev_all.items() if d["max_conf"] >= 0.90}
curr_090 = {s: d for s, d in curr_all.items() if d["max_conf"] >= 0.90}
prev_box_total = sum(d["box_count"] for d in prev_090.values())
rate = new_box_count / prev_box_total
converged = rate <= args.threshold          # threshold 默认 0.05（浮点，比率）
```

**与 Step7 的定义对比**：

| 维度 | Step7 `converge.py` | Step8 `converge.py` |
|:---|:---|:---|
| 单位 | **图片数** | **框数 + IoU 匹配** |
| 阈值 | `new_added < 30`（整数）或 `growth < 2%` | `rate ≤ 5%`（浮点比率） |
| 置信度过滤 | 无（纯文件名集合） | **`max_conf ≥ 0.90`** |
| IoU 逻辑 | **完全没有** | `iou_thr = 0.5` |

> **原文档只描述了 Step8 的那个**（`S8conclusion.md` §4.5），从未指出两者不同。

## 参数：训练超参数（全阶段固定）

`step8/tool/train.py` 的 `HP` 与 Step7 **完全相同**：

```python
HP = {
    "imgsz": 640, "epochs": 20, "batch": 16,     # batch 实际总被 OOM 降到 8
    "optimizer": "AdamW", "lr0": 1e-4,
    "momentum": 0.9, "weight_decay": 0.0005,
    "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
    "patience": 0,
}
# device=0, amp=False, workers=8
```

**磁盘实测的训练产物**（`step8/result/model_v12/train/`）：

```
args.yaml, results.csv, results.png, confusion_matrix*.png,
Box{F1,P,R}_curve.png, labels.jpg, train_batch0-2.jpg,
train_batch2730-2732.jpg,          ← 说明训练集有 ~2733 个 batch
val_batch0-2_{labels,pred}.jpg
weights/best.pt (49.6 MB), weights/last.pt (49.6 MB), weights/best.onnx (98.79 MB)
```

**`model_v12` 的实际 mAP50 = 0.9794**（`20260704_165356_train.log`）。

> **⚠️ 勘误**：`S8conclusion.md` §2.2 的表格称 "model_v12 ... mAP≈**0.956**"，"model_v9 mAP≈**0.971**"。实测训练日志：**model_v12 = 0.9794**，**model_v9 = 0.9706**。v9 对了，v12 错了 2.3 个点。
>
> `S8conclusion.md` §一 的一句话速查里也写"最优模型：model_v12（IoU=0.9, mAP≈0.956）"，同样错误。

## ⚠️ 环境与硬件陷阱（Step8 集中爆发）

| # | 陷阱 | 现象 | 规避 |
|:---|:---|:---|:---|
| **T1** | **僵尸进程累积** | 物理内存 28/32 GB，训练从 0.3 s/batch 恶化到 **30 s/batch** | 定期 `taskkill` 清理残留 `python.exe`，检查 `nvidia-smi` |
| **T2** | **硬链接导致 `dataN/val` 为空** | YOLO 报 "No images found in val" | 构建 data 时用 `shutil.copy2` 而非 `os.link` |
| **T3** | **训练极慢** | 第一个 epoch 高达 80 s/batch | 确保独享 GPU |
| **T4** | **OOM** | CUDA memory allocation failure（每个 epoch 都触发） | 清理 GPU 后重试，或降 batch（**但会静默丢参数**） |
| **T5** | **Resume 失败** | 恢复 checkpoint 时 CUDA OOM | **不 resume**，直接用 `last.pt` 作为 `weights` 参数重新训练 |
| **T6** | **种子覆盖导致模型新增无效** | 种子覆盖几乎全部图片 | 只在非种子图片上评估模型贡献 |
| **T7** | **过采样命名冲突** | `_r` 后缀文件被反复追加（`data10` 达 7240 张） | `S8计划` 声称用 `defaultdict(int)` —— **该修复不在任何现存脚本中** |
| **T8** | **LabelMe JSON 保存位置** | JSON 与图片同目录 | 符合 LabelMe 习惯 |

## 方法学总结

Step8 投入了约 15 轮训练（每轮 20 epoch，`batch=8`）产出了**最有价值的副产品**（339 张人工多框标注 → `labelme_dataset12` 的 3713 框），但它的**实验结论不可信**：

| 问题 | 影响 |
|:---|:---|
| `--iou 0.9` 从未生效（OOM 分支漏传） | Phase 2 vs Phase 3 的对比不成立 |
| train/val 有 264 张重叠 | 验证 mAP50 虚高 |
| `data10` 过采样累积失控 | 该轮训练分布与其他轮不可比 |
| 950 个标签未进入任何 split | 数据浪费 |
| R9/R10/R13 无训练日志 | 15 轮的连续性无法验证 |
| 收敛判定与训练时序错位 | 收敛曲线与训练曲线不同步 |
| 过采样脚本丢失 | 无法复现 `data9`~`data15` |

## 结论

Step8 用三阶段 15 轮把单框模型推向多框感知：Phase 1 收敛时仅 111 张图有 2+ 框，Phase 2 引入 **339 张人工多框标注 + 多框图 ×2 过采样**后 R5 检出框从 2568 一次性涨到 4841、R9 峰值 6060 框（首次 7 框），R12 收敛得 `model_v12`（mAP50 **0.9794**，`labelme_dataset12` 3713 框）并被 Step9 选用。但方法学上不成立：**`--iou 0.9` 因 OOM 分支漏传在全部 15 轮从未生效**（所有 train 日志均为 `Batch used: 8`）且该参数本就被误解为 NMS 阈值；**`data12` 的 train ∩ val 有 264 张重叠**使验证 mAP50 虚高；`data10` 过采样累积到 7240 张；950 个标签未进入任何 split；过采样脚本已丢失无法复现。
