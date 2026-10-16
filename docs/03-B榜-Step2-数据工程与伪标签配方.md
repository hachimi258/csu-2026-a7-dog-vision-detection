# B 榜 Step2：数据工程与伪标签配方探索

> **归档说明**：本文由项目全程技术复盘按"成果主线"重组而成，对应原复盘文档的相关章节（源文件 `archive/26中国软件杯_项目全程复盘.md`）。
> **性质**：基于代码、日志、产物实测的复盘记录，**非该阶段原始源码**；文中的代码片段为**原样节选（excerpt）**，用于说明关键实现，可能省略了与本节无关的部分。

## 目的

1. **重构验证集**（20 张 → 30 张，时序均匀抽取）——解决 Step1 暴露的评估不可靠问题
2. **搜索最优伪标签配方**——置信度阈值 × 加权方式的二维搜索
3. **把所有数据固化成唯一权威入口** `final_data/`

## 实现：流水线拆解

### ① `infer.py`：一次推理、离线调阈值

Faster R-CNN 对 2827 张推理，**保存原始框不设阈值**，结果序列化为 `pre_data/raw_predictions.pkl`：

```python
results[stem] = {'boxes': Nx4 float32, 'scores': N float32, 'labels': N int64}
pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
```

| 参数 | 值 |
|:---|:---|
| 模型 | `step1/weights/model_best.pth`（硬要求，缺失即中止） |
| batch_size | 4 |
| num_workers | 0 |
| 阈值 | **无** |

> 这是重要的工程决策：**推理一次、离线调阈值**。避免了每换一个阈值就重新推理 2827 张。日志：`20260625_200544_infer.log` 记录"推理 2827 张"。

### ② `gen_labels.py`：从 pkl 生成多阈值 XML

```python
thresholds = [0.99, 0.95, 0.90, 0.80, 0.75]
out_dir = data_root / f"thr_{int(thr*100):03d}"
for img_id, pred in raw.items():
    areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    max_idx = areas.argmax()                 # ⭐ 每图只取「面积最大」的框
    max_score = scores[max_idx]
    if max_score >= thr:
        # 坐标硬裁剪到 [0,639]×[0,479]
        box = boxes[max_idx]
        labels[max_idx] → class_name（缺失时默认 "firebig"）
        写 XML 到 data/thr_{int(thr*100):03d}/{img_id}.xml
# 同时输出 conf_map.json：{img_id: max_score}  用于 Exp-D2 逐框置信度加权
# 同时输出 split.json：sample100 固定 80/20 划分（seed=42）
```

**这一步定义了整个项目的核心不变式**：**任务被简化为"单类 + 每图一个最大面积框"的定位问题**。`max_idx`/`max_score` 是从**所有**预测框（不分类别）算出来的。

**⚠️ 硬编码 640×480**：坐标裁剪用 `min(x1, 639)` / `min(y1, 479)`，`create_xml()` 的默认 `width=640, height=480`。**脚本从不打开图片检查尺寸**（它只有 pkl，没有图像尺寸信息）。若某张 capture 尺寸不同，框会被裁到错误的坐标系且 XML 的 `<size>` 会撒谎。这只是因为 `captures` 恰好全是 640×480 才没出事。

### ⑤ 实际产出的阈值目录规模（磁盘实测）

| 阈值目录 | XML 数 | 占 2827 张的比例 |
|:---|:---|:---|
| `thr_099` | **1007** | 35.6% |
| `thr_095` | **1275** | 45.1% |
| `thr_090` | **1386** | 49.0% |
| `thr_080` | **1488** | 52.6% |
| `thr_075` | **1516** | 53.6% |

**只有约 54% 的 capture 在 ≥0.75 时产生了任何检测**（1311 张完全无检出）—— 这直接反映了"100 张训练出来的 Faster R-CNN 太弱"加上"只取最大框"规则的双重后果。

阈值目录是**嵌套的**（`thr_099 ⊂ thr_095 ⊂ thr_090 ⊂ thr_080 ⊂ thr_075`），这正是重采样方案可以叠加的原因。

### ③ `train_exp.py`（10 轮快速实验）：Faster R-CNN 的 8 组消融

固定变量：Faster R-CNN FPN-v2 / SGD lr=0.001 momentum=0.9 wd=0.0005 / StepLR(30, 0.1) / batch=2 训练、1 验证 / **每次实验都从 COCO 权重重新开始**（不从 step1 热启动）/ 验证集恒为 `sample100` 的 20 张。

| 实验 | XML 来源 | 训练 ID | 加权损失 |
|:---|:---|:---|:---|
| `Baseline` | `sample100` | `split.json` 的 80 张 | 否 |
| `Exp-thr_099` | `thr_099` | 目录内全部有效 ID | 否 |
| `Exp-thr_095` | `thr_095` | 同上 | 否 |
| `Exp-thr_090` | `thr_090` | 同上 | 否 |
| `Exp-thr_080` | `thr_080` | 同上 | 否 |
| `Exp-thr_075` | `thr_075` | 同上 | 否 |
| **`Exp-D1_sampling`** | `thr_080` | **`ids_095 * 3 + ids_080`** | 否 |
| `Exp-D2_weighted` | `thr_080` | `ids_080` | **是** |

> `Exp-D1_sampling` 是**在 Python 列表里重复 ID**（不是复制文件），DataLoader 会在每个 epoch 把同一张图喂 3 次。这就是"高置信度过采样"的实现机制。

**结果（`20260625_202804_train_exp.log`）**：

| 实验 | 最佳 F1 | 最佳 epoch |
|:---|:---|:---|
| **Exp-D1_sampling** | **0.8649** | 8 |
| Exp-thr_090 | 0.8235 | 8 |
| Baseline（纯真实 80 张） | 0.7692 | 8 |

> **结论 1**：伪标签有效（0.7692 → 0.8649，**+9.6 个点**）。
> **结论 2**：`高置信度×3 过采样` 优于等权 → **D1 配方胜出**。

**⚠️ 但 8 组实验只有 3 组真正跑完**。`step2/exp_weights/` 磁盘实测只有 9 个 `.pth`：`Baseline_best/final`、`Exp-D1_sampling_best/final`、`Exp-thr_090_best/final`、`Exp2-A_best`、`Exp2-B_best`、`Exp2-E_best`。

即 **`Exp-thr_099` / `Exp-thr_095` / `Exp-thr_080` / `Exp-thr_075` / `Exp-D2_weighted` 从未完成训练**——它们只是配置里存在。所谓的"8 组消融网格"在实践中是**3 组快速筛选**。

### ④ `train_exp2.py`（30 轮深度实验）：8 组实验、3 个启动档位

所有实验共享 `epochs=30, lr=0.001, CosineAnnealingLR, warmup=2`：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
# get_scheduler(warmup_epochs=2):
#   CosineAnnealingLR(optimizer, T_max=28)
#   LinearLR(optimizer, start_factor=0.1, total_iters=2)
#   SequentialLR([warmup, cosine], milestones=[2])
# 提前停止 patience = 5（基于验证 F1）
# 每个 epoch 都评估（不像 train_exp.py 按间隔）
```

| 实验 | 描述 | pseudo_thr | replicate | extra_thr | 区分配置 |
|:---|:---|:---|:---|:---|:---|
| `Exp2-A` | 30 轮 + 余弦退火 | 0.95 | 3 | 0.80 | 本档基线 |
| `Exp2-B` | 标签平滑 0.1 | 0.95 | 3 | 0.80 | `label_smoothing: 0.1` |
| `Exp2-C-2x` | 复制倍数消融 | 0.95 | **2** | 0.80 | ×2 |
| `Exp2-C-4x` | 同上 | 0.95 | **4** | 0.80 | ×4 |
| `Exp2-C-5x` | 同上 | 0.95 | **5** | 0.80 | ×5 |
| `Exp2-D-085` | 混合阈值 | 0.95 | 3 | **0.85** | extra 0.85 |
| `Exp2-D-090` | 混合阈值 | **0.90** | **2** | 0.80 | 0.90×2 + 0.80 |
| `Exp2-E` | 伪标签权重衰减 1.0→0.3 | 0.95 | 3 | 0.80 | `pseudo_weight_decay` |

档位：`TIER1=[A,B,E]`、`TIER2=[C-2x,C-4x]`、`TIER3=[C-5x,D-085,D-090]`。**实际只跑了 TIER1**（`--tier 1`，见日志 `python train_exp2.py --tier 1`）。

**结果（`20260626_015741_train_exp2.log`）**：

| 实验 | 最佳 F1 | 最佳 epoch |
|:---|:---|:---|
| **Exp2-A** | **0.8718** | **1** |
| Exp2-E | 0.8649 | 1 |
| Exp2-B | 0.7879 | 5 |

Exp2-A 逐轮轨迹：

```
E1 : Loss 0.0349  F1 0.8718  ← 峰值
E2 : Loss 0.0438  F1 0.8000  (骤降 8.2%)
E3 : Loss 0.0445  F1 0.8000
E4 : Loss 0.0366  F1 0.8000
E5 : Loss 0.0324  F1 0.7059
E6 : Loss 0.0226  F1 0.7647  → 早停
```

> **结论 3（重要）**：**最佳性能出现在第 1 个 epoch**，之后断崖式下跌。loss 持续下降但 F1 下降 ⇒ 模型在第一个 epoch 就跨过了泛化最优点。
> **结论 4**：标签平滑在本数据集上**有害**（0.7879 vs 0.8718）。
> **结论 5**：学习率 1e-3 量级过大，"第一步迈得太远"。后续必须降到 1e-4 量级。

## ⚠️⚠️ 三个"被记录但完全无效"的实验（原文档的核心结论建立在它们之上）

这是整个 Step2 最严重的问题。

| 实验 | 文件 | 为什么不生效 |
|:---|:---|:---|
| **`Exp-D2_weighted`** | `train_exp.py` | 全局变量 `_current_weights` **从未被赋值**。代码只在 `weighted_fastrcnn_loss` 里 `global _current_weights` 然后**只读**，所以 `'_current_weights' in globals()` 恒为 `False`，`weights` 恒为 `None`，回退到 `torch.ones_like(labels)` —— **加权损失与无权损失数值完全相同**。`VOCDataset` 认真地把 `target['weights'] = confs` 塞进 target，但**训练循环从不读它**。<br>**⇒ `Exp-D2_weighted` 是 `Exp-thr_080` 的重复实验。**<br>即使 `_current_weights` 被填上，`target['weights']` 是**按原始标注**粒度，而 `labels`/`box_regression` 是**按采样后的 RoI** 粒度（`sample_positive`/`sample_negative` 会重排和复制），`weights[pos]` 的索引是形状/顺序错配的。 |
| **`Exp2-B`（标签平滑 0.1）** | `train_exp2.py` | `label_smoothing` 只写在 `EXPERIMENTS` 配置字典里，**从未传给任何损失函数**。torchvision 内部的 `fastrcnn_loss` 调用 `F.cross_entropy(..., reduction="sum") / N`，**没有 `label_smoothing` 参数**。<br>**⇒ `Exp2-B` 与 `Exp2-A` 功能完全相同。** |
| **`Exp2-E`（伪标签权重衰减）** | `train_exp2.py` | `pseudo_w = get_pseudo_weight(epoch, train_cfg)` 每个 epoch 都算，但**只被插值进日志字符串**（`f"...(pseudo_weight={pseudo_w:.2f})"`）。它从不乘任何损失项、不影响采样器、不改数据集权重。<br>**⇒ `Exp2-E` 与 `Exp2-A` 功能完全相同。** |

**后果**：

- `S2conclusion.md` 的结论 **F3「标签平滑（Label Smoothing=0.1）在本数据集上有害」完全没有依据**——它比较的是两个相同的实验，0.7879 vs 0.8718 的差异纯属训练随机性
- 表 F1 行"Exp2-A（30 轮）Epoch 1 达 0.8718"与"Exp2-E 同模式"——**Exp2-E 就是 Exp2-A**，所以"同模式"是同义反复
- `exp_weights/` 里恰好只有 `Exp2-A_best.pth` / `Exp2-B_best.pth` / `Exp2-E_best.pth` 三个文件，而按上面的分析**这三个应该产出几乎相同的 F1**——这正是"三个实验一致，所以结论稳健"这种**极具误导性**的结果形态

## 实现：验证集重划分（`divide.py`）

**动机**：`sample100` 是**视频连续抽帧**，相邻帧高度相似。随机划分会让训练/验证集中出现近似重复帧，评估虚高。

**机制（精确）**：

```python
real_img_paths.sort()                                        # 文件名排序 = 时序
val_indices = np.linspace(0, total_real - 1, VAL_COUNT, dtype=int).tolist()
train_indices = sorted(set(range(total_real)) - set(val_indices))
```

`np.linspace(0, 99, 30, dtype=int)` 得到**均匀间隔**的索引：

```
[0, 3, 6, 10, 13, 17, 20, 23, 27, 30, 34, 37, 40, 44, 47,
 51, 54, 58, 61, 64, 68, 71, 75, 78, 81, 85, 88, 92, 95, 99]
```

训练集为剩余 70 张。**这个 30 张验证集此后永久固定，不再更改。**

日志（`divide_final_data.log`）：

```
真实样本总数: 100
清单已生成: real_train.txt (共 70 行)
清单已生成: real_val.txt   (共 30 行)
验证集抽取索引 (0~99): [0, 3, 6, 10, ...]
```

## ⚠️ 两套互不兼容的验证集并存

| 验证集 | 划分方式 | 规模 | 使用方 |
|:---|:---|:---|:---|
| `step2/pre_data/split.json` | `random.seed(42)` + `random.shuffle`，80/20 | 20 张 | **Step2 全部 Faster R-CNN 实验** |
| `step2/final_data/splits/real_val.txt` | 文件名排序 + `np.linspace(0,99,30)` | 30 张 | **Step3 / Step4 全部 YOLO 实验** |

`split.json` 的 20 张（磁盘实测）：`sample_065, sample_030, sample_028, sample_089, sample_098, sample_005, sample_055, sample_076, sample_012, sample_070, sample_087, sample_014, sample_018, sample_029, sample_032, sample_036, sample_095, sample_004, sample_015, sample_082`

> **⚠️ 这意味着 Step2 的 F1 与 Step3/Step4 的 mAP50 是在不同验证集上测的，任何跨框架的"前后对比"都被混淆了**，而且没有任何脚本做过调和。

## 数据固化：`final_data/`

```
step2/final_data/
├── images/                                 # 2827 张（物理拷贝）
├── annotations/
│   ├── real/                               # 100 个真实 XML
│   └── pseudo/
│       ├── thr_095/                        # 1275 个 XML
│       └── thr_080/                        # 1488 个 XML
├── splits/
│   ├── real_train.txt                      # 70 行：<图片绝对路径> <XML绝对路径>
│   ├── real_val.txt                        # 30 行（唯一验证标准）
│   └── pseudo_mixed_optimal.txt            # 5313 行
└── metadata.json
```

**只保留 5 个阈值目录中的 2 个**（0.95 与 0.80）——一次有意的收窄。

**最优配方（`pseudo_mixed_optimal.txt`）**：

```python
mixed_pairs = pseudo_095_pairs * 3 + pseudo_080_pairs
```

```
最优数据集 = (thr_095 的图片) × 3 + (thr_080 的图片) × 1
           = 1275 × 3  +  1488 × 1
           = 3825      +  1488      = 5313 行
```

磁盘实测：`real_train.txt = 70` 行、`real_val.txt = 30` 行、`pseudo_mixed_optimal.txt = 5313` 行 ✅ —— 算术精确对上。

`metadata.json` 记录了 `images_count`、真实样本数、`val_indices`、`val_files`、伪标签数、`mixed_formula: "thr_095 x 3 + thr_080 x 1"`、三个清单路径。

## ⚠️ 勘误：1275 vs 1887 的分母问题

| 来源 | thr≥0.95 的 XML 数 |
|:---|:---|
| Step1 `annotate.log`（阈值 0.95 直出） | **1887**（框数） |
| Step2 `final_data/annotations/pseudo/thr_095/` | **1275**（图片数） |

**原因**：Step1 把模型**所有 ≥0.95 的框**都写进 XML（一图可能多个 object）；Step2 的 `gen_labels.py` **每图只保留面积最大的那一个框**（`areas.argmax()`），所以 1887 框 → 1275 张图。

**这个差异恰好符合赛制**（每图只需 1 个框），但意味着：

- 原文档中"1488 张有目标 / 1339 张无目标"（Step1 口径）与"thr_095 共 1275 个"（Step2 口径）**是两个不同分母**，不是矛盾，但原文没说清
- Step2 的 1275 + 1488 说明：**阈值的实际语义变成了"面积最大框的置信度"**，而不是"任意框的置信度"。这个语义变化是合理的，但没有被声明

## 结论

Step2 确立了整个项目的核心不变式——"单类 + 每图一个最大面积框"，并以 `thr_095 × 3 + thr_080 × 1 = 5313` 行固化出唯一权威入口 `final_data/`；验证集也从 20 张重划为 30 张时序均匀抽取。但 **`Exp-D2_weighted`、`Exp2-B`、`Exp2-E` 三个实验在代码层面与对照组功能完全相同**，原文档建立在其上的"加权有效""标签平滑有害""伪权重衰减同模式"三条结论均无依据；同时 Step2（20 张 `split.json`）与 Step3/Step4（30 张 `real_val.txt`）的两套验证集并存，使跨框架的前后对比被混淆。
