# B 榜 Step4：超参数调优（优化器 × 学习率网格）

> **归档说明**：本文由项目全程技术复盘按"成果主线"重组而成，对应原复盘文档的相关章节（源文件 `archive/26中国软件杯_项目全程复盘.md`）。
> **性质**：基于代码、日志、产物实测的复盘记录，**非该阶段原始源码**；文中的代码片段为**原样节选（excerpt）**，用于说明关键实现，可能省略了与本节无关的部分。

## 目的

在固定数据集（`yolo_dataset_fixed`）上对候选模型搜索**优化器 × 学习率**组合，产出精度最高的权重。

## 实现：实验设计（`train1.py`）

```python
models      = ["yolov8m", "yolo11m"]
optimizers  = ["SGD", "AdamW"]
lrs         = [1e-4, 5e-4]
# 8 组组合，每组 epochs=5, imgsz=640, batch=16 (OOM→8)
```

**初始化方式**：`find_step3_best(model_key)` 在 `step3/result/20*` 中取最新目录，用**子串匹配**找 `best.pt`，然后 `YOLO(pretrain_path)`。代码注释精确说明了意图：**"仅加载权重，优化器自动重置"** —— 加载原始权重（而非训练 checkpoint）会丢弃 step3 的 SGD 动量与调度器状态，让每个网格单元都从新优化器开始。这是跑这类消融的正确做法。

## 参数：训练超参数（`train1.py`）

| 参数 | 值 |
|:---|:---|
| `epochs` | **5** |
| `imgsz` | **640** |
| `batch` | **16**（OOM → 8） |
| `patience` | **`epochs`** = 5 → **等效禁用早停**（`patience` 等于总轮数即保证跑满） |
| `save_period` | **1**（每轮存 checkpoint） |
| `seed` | **42** |
| `lr0` | **1e-4 或 5e-4** |
| `optimizer` | **`'SGD'` 或 `'AdamW'`** |
| `weight_decay` | **SGD → 0.0005；AdamW → 0.01**（AdaM 家族用更大 wd，是正确的实践） |
| `momentum` | **SGD → 0.937；AdamW → `None`** ← **BUG** |
| `device` / `workers` | 0 / 4 |

**目录命名**：

```python
save_name = f"{model_key}_{opt}_{lr:.0e}".replace('e-0', 'e-')
# yolov8m_SGD_1e-4  （{lr:.0e} 得 "1e-04"，replace 归一化为 "1e-4"）
save_dir  = os.path.join(MODEL_DIR, ts_root, save_name)
```

磁盘验证：`step5/export.py` 硬编码了 `step5\best1\yolov8m_AdamW_1e-4\train\weights\best.pt` —— **证实网格的赢家是 YOLOv8m + AdamW + lr 1e-4**。

**⚠️ 目录层级偏差**：`train1.py` 里 `ts_root` 是在 `run_exp()` **内部**计算的，所以 8 个格子落在 **8 个不同的时间戳目录**里，而不是一个运行级目录。`train1_1.py` 通过把 `root_ts` 提到 `main()` 修复了这个问题。

## ⚠️ `momentum=None` bug

```python
momentum=0.937 if opt=='SGD' else None,
```

对 AdamW 传 `momentum=None` 对 Ultralytics 的优化器构建器是非法值（它期望 float，`None` 不是合法的 beta1 覆盖）。这个 bug **没有硬崩溃**（所以实验还是跑出了权重），但是潜在的失败点。

**`train1_1.py` 显式修复**：

```python
momentum_val = 0.937 if opt == 'SGD' else 0.9
# 注释：====== 修正：对于AdamW，momentum不能为None，设为0.9（默认beta1） ======
```

**同时新增 `amp=False`**（主路径与 OOM 重试路径都加）。这**关闭了自动混合精度** —— 一个明显的环境适配 hack（很可能是为了规避 RTX 5060 + 夜间版 torch 组合下的 NaN/溢出）。代价是速度，收益是消除一整类不稳定。

> **注意**：`amp=False` **只出现在 `train1_1.py`**。`step3/train1*.py` 和 `step4/train1.py` 都跑在 Ultralytics 默认 AMP 开的状态下。所以 Step3 与 Step4 的数值表现**不可直接比较**。

## ⚠️ 日志解析式断点续传（`train1_1.py`）

作者用**解析上一次运行的日志文件**来判断哪些实验已完成 —— 一个不依赖文件系统的进度追踪器。

```python
start_pattern  = re.compile(r'>>> 开始 (\w+) \| (\w+) \| lr=([\d.eE+-]+)')
finish_pattern = re.compile(r'(✅|⚠️)\s+(\S+)\s+(?:最佳mAP50 = ([\d.]+)|未生成best\.pt)')
```

两遍扫描：先收集所有 `started` 三元组，再收集 `completed`（带 map50 值，或 `None` 表示 `⚠️ … 未生成best.pt`）。`unfinished = started - set(completed)`，实际运行列表 = `all_combinations - completed`。

从完成行恢复 key：

```python
parts = name.split('_')
model, opt = parts[0], parts[1]
lr_part = '_'.join(parts[2:])          # "1e-4"
lr = float(lr_part.replace('e-','e-')) # ← no-op replace（作者大概想归一化 e-04 → e-4）
```

**只从 `*_train1.log` 恢复**，所以 `train1_1.py` 自己（日志名 `*_train1_1.log`）无法被续传 —— 一个有意设计的一跳链。

> **⚠️ 极度脆弱**：这个续传逻辑依赖日志字符串的精确 emoji + 空格（`✅ {save_name} 最佳mAP50 = {map50:.4f}`）。任何日志格式改动都会**静默地**让 `completed` 变空，从而**全量重跑**。这是一个"沉默失效"的设计。

## 结果（`20260628_115147_test.log`）

`test.py` 暴力扫描所有 `.pt` 并重新验证，产出排名：

| 排名 | 优化器 | lr | mAP50 | 权重目录（时间戳） |
|:---:|:---|:---|:---:|:---|
| 🥇 | **AdamW** | **1e-4** | **0.9888** | `20260627_232146` |
| 🥈 | SGD | 5e-4 | 0.9759 | `20260627_111418` |
| 🥉 | AdamW | 5e-4 | 0.9681 | `20260627_232146` |
| 4 | AdamW | 1e-4（同参数重跑） | 0.9536 | `20260627_155216` |
| 5 | SGD | 1e-4 | 0.9505 | `20260627_101056` |

**日志中的反复失败**（`⚠️ 未生成best.pt`）：

```
[2026-06-27 11:14:17] ⚠️ yolov8m_SGD_1e-4 未生成best.pt
[2026-06-27 11:33:07] ⚠️ yolov8m_SGD_5e-4 未生成best.pt
[2026-06-28 06:10:51] ⚠️ yolov8m_AdamW_1e-4 未生成best.pt    ← 跑了 6.8 小时却"白跑"
[2026-06-28 11:26:36] ⚠️ yolov8m_AdamW_5e-4 未生成best.pt
```

> **解读**：日志的"未生成 best.pt"是**脚本自己的路径判断出错**（`project=save_dir, name=""` 在不同 Ultralytics 版本下产出 `save_dir/weights/` 或 `save_dir/train/weights/`），而 Ultralytics 其实正常写了权重——这正是 `test.py` 要**同时查找 `weights/` 和 `train/weights/`** 的原因。所以结果表仍然可用。

## ⚠️⚠️ Step4 最严重的问题：同参数两次运行差 3.5 个点

**第 1 名（0.9888）和第 4 名（0.9536）是同一组超参数（AdamW + 1e-4）的两次运行，差了 3.5 个点。**

| 目录 | mAP50 |
|:---|:---|
| `model/20260627_232146/yolov8m_AdamW_1e-4/` | **0.9888** |
| `model/20260627_155216/yolov8m_AdamW_1e-4/` | **0.9536** |

**原因分析**：

1. **验证集只有 30 张** ⇒ 单个样本翻转就能改变 **3.33 个点**（1/30）
2. 两次运行的差异（3.5 点 ≈ 1 个样本）**完全在噪声范围内**
3. 训练本身存在未记录的随机性（虽然 `seed=42` 设了，但 CUDA 非确定性算子、数据加载顺序、以及两次运行之间 `amp` 设置不同都会影响）

> **结论**：**Step4 的整个超参数排名是噪声**。"AdamW+1e-4 最优"这个结论**没有统计依据**。
>
> **但有趣的是**：最终提交的模型确实用了这个组合（`step5/best1/yolov8m_AdamW_1e-4/`），而它在 Step6 的平台实测中给出了 F1=0.37529。**这个选择是否比 SGD+5e-4 更好，从未被验证过。**
>
> **正确的做法**：至少 100~200 张验证集，或 k-fold 交叉验证。Step7 之后就达到了 320 张（单样本影响 0.31 点），但那时模型已经选定了。

## Step4 耗时的实测修正

日志时间戳（`20260627_232145_train1_1.log`）：

```
23:21:46 开始 yolov8m_AdamW_1e-4
06:10:51 结束（标记未生成 best.pt）       → 6.8 小时
06:10:51 开始 yolov8m_AdamW_5e-4
11:26:36 结束                              → 5.3 小时
```

`20260627_101056_train1.log`：

```
10:10:56 开始 yolov8m_SGD_1e-4
11:14:17 结束（63 分钟）
11:14:18 开始 yolov8m_SGD_5e-4
11:33:07 结束（19 分钟）
```

**⚠️ 勘误**：`S4conclusion.md` 称 "AdamW 训练耗时约 5~6.5 小时/5 轮"（实测 6.8 小时），称 SGD+5e-4 "训练耗时极短，仅 **0.3 小时**"（实测 19 分钟 = 0.32 小时 ✅ 这个对，但同脚本跑的 SGD+1e-4 用了 63 分钟）。

**最终选择**：AdamW + 1e-4（0.9888），权重复制到 `step5/best1/yolov8m_AdamW_1e-4/train/weights/best.pt`。

**选择理由**（原文档）：推理速度与优化器无关，只在训练时体现差异，所以选精度最高的无速度风险。（这个推理本身是对的，只是"精度最高"这个前提不可靠。）

## 结论

Step4 用 8 组"优化器 × lr"网格（`epochs=5`，`save_period=1`，`patience=epochs`）选出 AdamW + 1e-4 的 0.9888 作为最终权重，但**同一组超参数的两次运行相差 3.5 个点（0.9888 vs 0.9536），而 30 张验证集下单样本翻转即值 3.33 个点**——因此整个超参数排名落在噪声范围内，"AdamW+1e-4 最优"没有统计依据。此外 `train1.py` 的 `momentum=None`（AdamW）是潜在非法值，`train1_1.py` 修复为 0.9 并新增 `amp=False`（导致 Step3 与 Step4 数值表现不可直接比较）。
