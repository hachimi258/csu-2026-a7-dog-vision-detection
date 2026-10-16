# B 榜 Step1：Faster R-CNN 自动标注器（教师模型与首轮伪标签）

> **归档说明**：本文由项目全程技术复盘按"成果主线"重组而成，对应原复盘文档的相关章节（源文件 `archive/26中国软件杯_项目全程复盘.md`）。
> **性质**：基于代码、日志、产物实测的复盘记录，**非该阶段原始源码**；文中的代码片段为**原样节选（excerpt）**，用于说明关键实现，可能省略了与本节无关的部分。

## 目的

只有 100 张真实标注（`sample100`）可用。Step1 训练一个"够用"的检测器，用它给 2827 张无标注图打伪标签，作为整个自训练（self-training）链路的起点。

**关键定位**：Step1 的模型是**教师（teacher）**，不是最终提交模型。最终提交用的是 step3/step4 训练的 YOLOv8m。

## ⚠️ 架构勘误：不是 YOLO，是 Faster R-CNN

`S1-5conclusion.md` 第 55 行称"基于 `sample100` 训练 **YOLOv5-like** 模型"，**这是错误的**。

代码证据（`step1/train.py`）：

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, \
    FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

日志中出现的三种 loss 正是 Faster R-CNN 的损失：

```
Epoch 10/150 | Loss: 0.0112 (cls: 0.0053, box: 0.0053, rpn: 0.0001)
                                          ↑        ↑          ↑
                          loss_classifier  loss_box_reg  loss_rpn_box_reg
```

| torchvision 损失键 | 来源 | 是否被记录 |
|:---|:---|:---|
| `loss_classifier` | RoI 头 —— Fast R-CNN 分类 | ✅ `cls` |
| `loss_box_reg` | RoI 头 —— Fast R-CNN 框回归 | ✅ `box` |
| `loss_objectness` | RPN —— 二分类 objectness | ❌ **从不记录** |
| `loss_rpn_box_reg` | RPN —— 提议框回归 | ✅ `rpn` |

> **副作用**：打印的"总损失"是四项之和，但只显示三项，所以日志里 `总损失 > cls+box+rpn`。这是一个日志不一致。

**架构细节**：

- **Backbone**：ResNet-50 + FPN，**v2 变体**（torchvision 改进版配方）
- **Head**：标准 `FasterRCNN` —— RPN + `RoIHeads`（MultiScaleRoIAlign + box head + `FastRCNNPredictor`）
- **Anchor**：**全项目没有任何手工定义 anchor**。使用 torchvision 默认值：5 种尺度 `(32,64,128,256,512)` × 3 种长宽比 `(0.5,1.0,2.0)` = 每层每位置 15 个 anchor，覆盖 5 个 FPN 层
- **类别数**：`num_classes = 1 + 1 = 2`（背景 + `firebig`），`FastRCNNPredictor(in_features=1024, num_classes=2)`
- **输入处理**：全程 `transform=None`，只有 `transforms.functional.to_tensor` → **[0,1] float，无归一化、无 resize、无增强**。torchvision 内部的 `GeneralizedRCNNTransform` 做 `min_size=800, max_size=1333` 缩放。对 640×480 的图得到 **1067×800**（scale = 800/480 ≈ 1.667）。
- **类别发现**：从数据里读（`_get_classes()` 收集所有 `<name>` 文本排序），背景固定 index 0

## 参数

### `train.py`（首次训练，50 epoch）

| 项 | 值 |
|:---|:---|
| Epochs | **50** |
| Batch size | **2** |
| Optimizer | **SGD**, momentum=0.9, weight_decay=0.0005 |
| Learning rate | **0.005** |
| LR scheduler | **StepLR(step_size=5, gamma=0.1)** → 5e-3 / 5e-4 / 5e-5 / … |
| 训练/验证划分 | **无——全部 100 张都用于训练** |
| `num_workers` | **0**（Windows 安全，禁用多进程） |
| Seed | **未设置** |
| Checkpoint | 仅结束时保存 |
| 总迭代数 | 50 epoch × (100/2) = **2500 次** |

### `train2.py`（续训，150 epoch）

| 项 | 值 |
|:---|:---|
| Extra epochs | **150**（每 10 epoch 评估一次） |
| Batch size | 训练 **2** / 验证 **1** |
| Optimizer | **SGD**, momentum=0.9, weight_decay=0.0005 |
| Learning rate | **0.001** |
| LR scheduler | **StepLR(step_size=30, gamma=0.1)** |
| 验证比例 | **0.2**（`int(100*0.2)` = 20 张） |
| Seed | **`torch.manual_seed(42)`**（在 `random_split` 前立即设置） |
| 划分方式 | `torch.utils.data.random_split` |
| 指标 | **macro F1** @ IoU=0.5, conf=0.5 |
| Checkpoint | `model_epoch_{10..150}.pth`、`model_best.pth`、`model_final.pth` |

> **注意**：`train2.py` 用 `fasterrcnn_resnet50_fpn_v2(num_classes=num_classes)` **不带 COCO 权重**构造模型，然后 `load_state_dict(model.pth)`。即**这一步完全没有用 COCO 预训练**，模型完全由 `train.py` 的权重定义。
>
> 另外**从不保存 optimizer state**，所以"续训"实际上是"重启学习率调度"。`train2.py` 有意用 0.001 重新开始 `StepLR(step_size=30)`。

## ⚠️ 严重方法论缺陷：F1 评估跨图匹配

`train2.py` / `eval.py` / `train_exp.py` / `train_exp2.py` 共用的 `evaluate_f1()` 逻辑：

1. 模型 `eval()`，在验证集上 `no_grad()` 推理
2. **把整个验证集所有 GT 框和所有预测框（带分数）分别池化成两个扁平列表**（不记录来自哪张图）
3. 预测框按分数降序
4. 贪心匹配：每个预测框与**池中所有 GT 框**算 IoU，取最大；若 ≥0.5 且该 GT 未被匹配过，则记 TP
5. `tp = 分数≥0.5 的 TP 数`，`fp = 分数≥0.5 的非 TP 数`，`fn = n_gt - tp`

**致命问题**：GT 列表和预测列表**都没有绑定 image id**，匹配是在池化集合上做的，**所以 A 图的预测框可以匹配 B 图的真值框**。TP/FP 计数会任意膨胀或收缩。

**对照证据**：`step6/eval.py` 里作者**把这件事做对了**——它用 `if ge[0] != pe[0]: continue` 做了同图约束。所以 step1/step2 的跨图 bug 是真实缺陷，不是理解偏差。

> **结论**：**Step1 和 Step2 的所有 F1 数字都不可作为权威指标引用**。Step6 的 F1 是正确计算的。

`step1/eval.py` 还有额外的指标问题：

- **手写梯形左黎曼和 AP**（非 VOC 11 点、非全点插值），且**没有做 precision envelope（单调递减包络）**，系统性偏低
- 固定的 conf 阈值扫描存在**顺序缺陷**：一个 GT 框可能先被低分预测框"吃掉"（匹配时不看阈值），导致高分预测框无法再匹配，压低召回
- `micro_f1 = 2*Σ(R*P)/(ΣR+ΣP)` **不是真正的 micro-F1**（那是类内平均的调和平均），`macro_f1 = np.mean(f1s)` 才是正确的宏平均
- 若某类无 GT，强制 `ap=0.0, f1=0.0`（即使也没有预测），会拖低 mAP

## 结果

| 阶段 | 最佳验证 F1@0.5 | 说明 |
|:---|:---|:---|
| 首次训练 50 epoch | **0.9302** | loss 收敛至 ~0.015 |
| 续训 150 epoch | **0.9091**（epoch 10 之后持续下滑） | 最终 0.8333；**验证集仅 20 张导致剧烈震荡** |

续训过程中 F1 的完整轨迹（`20260625_180646_train2.log`）：

```
E10:0.9091  E20:0.8889  E30:0.8333  E40:0.8511  E50:0.8889  E60:0.8333
E70:0.8511  E80:0.8333  E90:0.8333 E100:0.8333 E110:0.8163 E120:0.8511
E130:0.8333 E140:0.8696 E150:0.8333
```

loss 从 0.0167 稳定降到 ~0.0059，但 F1 在 0.82~0.91 间随机跳动 ⇒ **验证集太小，指标不可用**。这一发现直接促成了 Step2 的验证集重划分。

> **但同时要打折**：这些 F1 本身受 4.4 节的跨图匹配缺陷污染。**即使趋势方向可信，"0.93"这个绝对数值也不可信。**

## 首轮伪标签生成（`step1/annotate.py`）

用 `model_best.pth` 对 2827 张 `captures` 推理：

| 参数 | 值 |
|:---|:---|
| `confidence_threshold` | **0.95** |
| `batch_size` | **4** |
| 模型选择优先级 | `["model_best.pth", "model_final.pth", "model.pth"]` —— 第一个存在的胜出 |
| 输出 | Pascal VOC XML → `step2/data/{stem}.xml`（扁平目录）<br>带框预览图 → `step2/view/{stem}.jpg`（质量 95，仅在 ≥1 个框时写） |

**重要行为**：`annotate.py` **总是**写 XML —— 即使 `objects == []` 也写一个空的 `<annotation>`（零个 `<object>` 子节点）。所以"无火"被编码成空 VOC 文件，而不是文件缺失。

**置信度分布**（共 4479 个检测框，`20260625_185833_annotate.log`，在阈值化**之前**收集所有原始分数）：

| 阈值 | 框数 | 占比 |
|:---|:---|:---|
| ≥ 0.50 | 2875 | 64.19% |
| ≥ 0.70 | 2633 | 58.79% |
| ≥ 0.80 | 2475 | 55.26% |
| ≥ 0.90 | 2184 | 48.76% |
| ≥ 0.95 | 1887 | 42.13% |
| ≥ 0.98 | 1554 | 34.70% |
| ≥ 0.99 | 1278 | 28.53% |

最高 0.9998，最低 0.0500，**平均 0.6496，中位数 0.8861**。

> **中位数 0.8861 ≫ 均值 0.6496** 说明分布严重左偏：大部分框置信度很高（>0.88），但有一批极低置信度的长尾。这个双峰结构是后续"分层阈值"策略的数据依据。

**⚠️ 诊断死胡同**：这个精心统计的置信度直方图**只被记录，从未被用来选阈值**。后续 `gen_labels.py` 用的阈值 `[0.99, 0.95, 0.90, 0.80, 0.75]` 是硬编码的。

**⚠️ 冗余**：`annotate.py` 写的 `step2/data/{stem}.xml` 与 `gen_labels.py` 写的 `step2/data/thr_095/` **内容重叠但互不引用**——后者是从 pkl 重新阈值化的，不是复用前者的 XML。

## 结论

Step1 的教师模型是 torchvision Faster R-CNN（ResNet50-FPN-v2，`num_classes=2`），50 epoch 首次训练后在 20 张验证集上得到最佳 F1 0.9302，续训 150 epoch 仅得 0.9091 且持续下滑；首轮伪标签以 `confidence_threshold=0.95` 覆盖 2827 张 `captures`。但**跨图匹配缺陷使 Step1/Step2 的所有 F1 数字都不能作为权威指标引用**，而这个由 20 张验证集造成的震荡直接促成了 Step2 的验证集重划分。
