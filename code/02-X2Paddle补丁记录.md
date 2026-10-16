# X2Paddle 补丁记录（含 5 个产物的版本对照）

> **归档说明**：本文由项目全程技术复盘按"成果主线"重组而成，数据来源为原复盘文档 §8.3 / §8.4 / §14.4（源文件 `archive/26中国软件杯_项目全程复盘.md`）。
> **性质**：基于代码与产物实测的复盘记录，**非原始源码**；代码片段为**原样节选（excerpt）**。
> **为什么单独成篇**：这是全项目**最容易重复踩坑、也最容易失传**的一处工程知识 —— Step6 手工打过并验证有效的补丁，在 Step9 重新转换时没有沿用，直接导致 Step9 只能 CPU 推理。

---

## 一、结论速览（TL;DR）

| 问题 | 结论 |
|:---|:---|
| 项目里有几个 `x2paddle_code.py`？ | **5 个**，分属**两种互不相同的版本** |
| 到底有几个补丁？ | **3 个**（1 个改 X2Paddle 源码，2 个改生成产物） |
| 补丁能自动生成吗？ | **不能**。"硬编码 slice 是补丁自动产生的"说法**不成立** |
| 重装 X2Paddle 会怎样？ | **丢失源码补丁 #1** |
| 最严重的后果 | **Step9 未应用补丁 #3 ⇒ Step9 提交包 CPU-only** |

---

## 二、5 个产物的版本对照

| 文件 | 行数 | `paddle.slice` 的 starts/ends | `paddle.cast` |
|:---|:---:|:---|:---|
| `step5/tool/x2paddle_output/` | 514 | **动态**：`starts=Constant_7_cast, ends=Mul_cast` | 10 个（含 **4 个 float32 cast** + int32） |
| **`step5/tool/x2paddle_v2_output/`** | 504 | **硬编码**：`starts=[0], ends=[2]` / `starts=[2], ends=[4]` | **0 个** |
| `step6/P2/x2paddle_512/` | 491 | **硬编码**：同上 | **0 个** |
| **`step6/P2/x2paddle_512_v2/`** | 491 | **硬编码**：同上 | **0 个** |
| `step9/model/x2paddle_output/` | 514 | **动态**：`starts=Constant_7_cast, ends=Mul_cast` | 5 个（1 int64 + 4 int32，**无 float32**） |

---

## 三、三个补丁的准确描述

| # | 补丁 | 应用目标 | 证据 |
|:---|:---|:---|:---|
| **1** | **MaxPool dilation 断言放宽** → `x2paddle/op_mapper/onnx2paddle/opset_legacy.py` ~L1980，允许 `dilation ∈ {None, [1], [1,1]}` | **X2Paddle 源码**（两个 conda 环境都要打） | 文档记录；不可从产物直接验证，但转换能成功说明必须 |
| **2** | **int64/float32 类型提升** → 算术运算前显式 `paddle.cast(..., 'float32')` | **只在 `step5/tool/x2paddle_output/`（v1，514 行）**，在 `_v2` 中被移除 | 该文件 L485-488 有 4 处 float32 cast |
| **3** | **DFL Slice 索引硬编码** → `starts=[0], ends=[2]` / `starts=[2], ends=[4]` | **`x2paddle_v2_output/`、`x2paddle_512/`、`x2paddle_512_v2/`** | 三者都是硬编码整数 |
| **3'** | **补丁 #3 未应用** | **`step9/model/x2paddle_output/`** | 动态 `starts=Constant_7_cast, ends=Mul_cast` ⚠️ |

### 补丁 1：MaxPool dilation（改 X2Paddle 源码）

**现象**：转换直接失败。

**根因**：YOLOv8 SPPF 使用 `MaxPool(dilation=[1,1])`，而 X2Paddle 只允许 `dilation=None`。

**修复**：修改 `x2paddle/op_mapper/onnx2paddle/opset_legacy.py` ~L1980，放宽断言。

> ⚠️ **两个 conda 环境都要打；重装 X2Paddle 会丢失。**

### 补丁 2：int64/float32 类型提升

**现象**：转换后代码在 Paddle 3.x 下算术运算报错。

**修复**：在 `add`/`divide`/`multiply` 前显式 `cast(..., 'float32')`。

`step5/tool/x2paddle_output/` 的 L485-488（真实代码）：

```python
Add_output_0   = paddle.add(x=paddle.cast(Gather_output_0, 'float32'),
                            y=paddle.cast(Constant_8, 'float32'))
Div_output_0   = paddle.divide(x=Add_output_0, y=paddle.cast(Constant_9, 'float32'))
Mul_output_0   = paddle.multiply(x=Div_output_0, y=paddle.cast(Constant_10, 'float32'))
Mul_1_output_0 = paddle.multiply(x=Div_output_0, y=paddle.cast(Constant_11, 'float32'))
```

> ⚠️ **此补丁只在 v1 产物里存在**，在 `_v2` 系（504/491 行）中已被移除。

### 补丁 3：DFL Slice 索引硬编码（最关键）

**现象**：同一模型 **CPU 推理正常，GPU 推理输出全零**。

**根因**：X2Paddle 生成的 DFL 切片用**动态索引**，在 GPU 上求值为非法值（`end=0, start=4`）。

**未修补版**（`step9/model/x2paddle_output/x2paddle_code.py`，514 行 v1）真实代码：

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

**修补版**（Step6 实际使用的 `x2paddle_512_v2/`）：

```python
Slice   = paddle.slice(input=dfl_Reshape_1_output_0, axes=[1], starts=[0], ends=[2])
Slice_1 = paddle.slice(input=dfl_Reshape_1_output_0, axes=[1], starts=[2], ends=[4])
```

**修复依据**：输出形状固定（`[1,4,8400]` / `[1,4,5376]`），所以 DFL 的 4 通道切分必然是两个 `[0,2)` / `[2,4)`。彻底消除风险，也让模型可在 GPU 上运行。

---

## 四、⚠️ 对原文档两处说法的修正

| 原文档说法 | 实测事实 |
|:---|:---|
| 补丁 #2 和 #3 是"**同一个文件的两处修改**"（`x2paddle_code.py` 约 L485-488 / L482-494） | 它们是**不同版本文件上的不同改动**：#2 只在 `step5/tool/x2paddle_output/`（514 行），#3 在 `_v2_output/`、`x2paddle_512/`、`x2paddle_512_v2/`（504/491 行） |
| "**26csresult 的 X2Paddle 已打源补丁，自动生成硬编码 `starts=[0], ends=[2]`**" | **不成立**。`step9/model/x2paddle_output/`（**同样 514 行、同样 26csresult 环境**）生成的是**动态**索引 ⇒ **硬编码是手工编辑的结果，不是补丁自动产生的** |

---

## 五、最严重的后果：Step9 是 CPU-only

Step9 的产物**两个补丁都没有应用**（无 float32 cast、slice 仍是动态），因此：

```python
# step9/predict.py
try:
    paddle.device.set_device('gpu'); config.enable_use_gpu(1000,0)   # 1000 MB（不是 2000）
except:
    paddle.device.set_device('cpu'); config.disable_gpu(); config.enable_mkldnn()
```

**Step9 的提交包只能 CPU 推理。** 而 Step6 的 P2 是 GPU 推理（`enable_use_gpu(2000,0)`），平台实测 58.28 FPS。

> **Step9 若真的提交，FPS 会是 CPU 水平（本地 eval 是 10.81 FPS）—— 远低于 20 门槛，会被强制判 0 分。**
>
> 而 Step9 的 `upload/submission.zip`（84.10 MB，07-05 07:23）**没有留下任何 FPS 记录**（`step9/log/` 为空）。

---

## 六、可执行的改进建议（来自复盘）

1. **把补丁做成可执行的后处理脚本**（如 `postprocess_x2paddle.py`），而不是手工编辑生成的产物 —— 手工编辑无法跨 step 传递。
2. **把"GPU 输出全零"做成验收测试**：转换后必须 CPU + GPU 双向跑一遍同一张图并比对输出。
3. **补丁 #1 记录成环境清单**：重装 X2Paddle 会丢失，必须能重建。
4. **转换环境必须匹配推理环境的 Paddle 版本**：PIR `.json` 格式在 Paddle 2.6.2 上会 segfault。

---

## 七、延伸阅读

- 补丁产生的完整上下文 → [../docs/06-B榜-Step5-PyTorch转Paddle与首次提交.md](../docs/06-B榜-Step5-PyTorch转Paddle与首次提交.md)
- 512 模型的重导出流程 → [../docs/07-B榜-Step6-速度优化.md](../docs/07-B榜-Step6-速度优化.md)
- Step9 的 CPU-only 后果 → [../docs/10-B榜-Step9-竖棍幻觉与可视化核验.md](../docs/10-B榜-Step9-竖棍幻觉与可视化核验.md)
