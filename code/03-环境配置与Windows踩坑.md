# 环境配置与 Windows/硬件踩坑清单

> **归档说明**：本文由项目全程技术复盘按"成果主线"重组而成，数据来源为原复盘文档 §2.2 / §14.5 / §15（源文件 `archive/26中国软件杯_项目全程复盘.md`）。
> **性质**：基于代码、日志、产物实测的复盘记录，**非原始源码**；代码片段为**原样节选（excerpt）**。

---

## 一、双 Conda 环境（关键设计）

| 环境 | Python | 关键组件 | 用途 |
|:---|:---|:---|:---|
| **26cnsoft** | 3.11.15 | PyTorch 2.10.0.dev20251026+cu128、TorchVision 0.25.0.dev20251028+cu128、Ultralytics 8.4.78、X2Paddle 1.6.0、Paddle 3.3.1(CPU)、ONNX | YOLO 训练 / GPU 推理 / ONNX 导出 |
| **26csresult** | 3.10 | Paddle 2.6.2(GPU)、cuDNN 8.6、CUDA 11.8 runtime、X2Paddle 1.6.0（已打补丁） | X2Paddle 转换生成 `.pdmodel`、本地 CPU 精度评估 |

**硬件**：NVIDIA RTX 5060 8GB（日志提及 RTX 5060 Ti，Compute Capability 12.0）。

### 环境搭建要点（`step1/setup_env.py`）

创建 conda 环境 `26cnsoft`（Python 3.11）→ 配置 pip 清华镜像 → 安装 numpy/jinja2/typing_extensions → **从本地 wheel 安装夜间版 PyTorch**（`torch-2.10.0.dev20251026+cu128-cp311-cp311-win_amd64.whl`、`torchvision-0.25.0.dev20251028+cu128-...whl`）→ 安装 pillow → 验证导入。

### `26csresult` 必须手动放置的 DLL

放在 `...\envs\26csresult\Lib\site-packages\paddle\libs\`：

- `cudnn64_8.dll` 等（来自 cuDNN 8.6.0 CUDA11 archive）
- `cudart64_110.dll`、`cublas64_11.dll`、`cublasLt64_11.dll`（CUDA 11.8 runtime）
- `zlibwapi.dll`

> **为什么必须双环境**：Paddle 2.6.2 生成的推理模型是旧格式 `.pdmodel`，能被竞赛平台加载；而 Paddle 3.3.1 的 X2Paddle 会输出 PIR `.json` 格式，在 Paddle 2.6.2 推理引擎上直接 **segfault**。这个"环境分工"是整个 B 榜工程链路的基石。

---

## 二、⚠️ 本地 GPU 完全不可用（贯穿全程）★★★

- **现象**：Paddle 2.6.2（CUDA 11.8 runtime）不支持 RTX 5060 Ti（CC 12.0），GPU 推理**输出全零**或 segfault
- **影响**：**本地无法测速**，FPS 只能靠平台反馈
- **后果**：Step7 的"预估 FPS 35–45"从未被验证，直接影响了后续路线决策

**Step5 的实测对照**：

| 测试场景 | 结果 |
|:---|:---|
| 本地 CPU 精度（30 张） | ✅ 30/30 检出，297 ms/张，FPS 3.4 |
| 本地 GPU（Paddle 2.6.2） | ⚠️ **输出全零** |

---

## 三、设备策略（完整代码）

```python
# B 榜推理：GPU 优先 + CPU 自动回退
try:
    config = Config(model_file, params)
    config.enable_use_gpu(2000, 0)          # Step9 用 1000
    config.disable_glog_info()
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)
except Exception:
    config = Config(model_file, params)
    config.disable_gpu()
    config.disable_glog_info()
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

# 本地 CPU 评估（step6/eval.py 与 step9/tool/*）
config.disable_gpu()
config.enable_mkldnn()
config.set_cpu_math_library_num_threads(4)   # step9 visualize.py
# step6/eval.py 额外用 config.enable_memory_optim()

# A 榜推理：固定 GPU + 关闭 IR 优化
config.enable_use_gpu(2000, 0)
config.switch_ir_optim(False)      # ← PP-YOLOE+ 与 IR 优化冲突
config.disable_glog_info()
config.enable_memory_optim()
config.switch_use_feed_fetch_ops(False)
```

---

## 四、Windows 专用 hack 汇总

| hack | 位置 | 原因 |
|:---|:---|:---|
| `num_workers=0` | Step1/Step2 所有 DataLoader | Windows 多进程问题 |
| `findstr` | `step1/setup_env.py` | Windows-only 命令 |
| `sys.stdout.reconfigure(encoding='utf-8')` | `step5/export.py` | emoji/中文日志让控制台崩溃 |
| `amp=False` | `step4/train1_1.py`、`step7/train.py`、`step8/train.py` | 规避 AMP 不稳定 / 避免下载 `yolo26n.pt` 卡住 |
| `np.frombuffer` + `cv2.imdecode` | `7.19/run_preview_v2.py`、`fix_bbox*.py` | `cv2.imread` 不支持中文路径 |
| 全项目 `encoding='utf-8'` | 所有文件句柄 | 默认编码是 GBK |
| `hardlink_or_copy()` | 所有数据集构建脚本 | `os.link` 跨卷失败 |

### 逐条说明

**① `num_workers=0`** —— Step1/Step2 所有 DataLoader。Windows 多进程问题。

**② `sys.stdout.reconfigure(encoding='utf-8')`**（`step5/export.py`）：

```python
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')   # 否则 emoji/中文日志会让 Windows 控制台崩溃
```

**③ `amp=False`** —— 训练启动时会反复 curl 超时下载 `yolo26n.pt`，每次约 3 分钟；`amp=False` 跳过 AMP check。

> ⚠️ **副作用**：`amp=False` 只在 `step4/train1_1.py` / `step7` / `step8` 出现，Step3 与 Step4 首版是默认 AMP 开 ⇒ **它们的数值表现不可直接比较**。

**④ `cv2.imread` 中文路径**（`7.19/`）：

```python
def imread_safe(path):
    """cv2.imread 不支持中文路径的替代方案"""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
```

`cv2.imread` 在 Windows 上**无法读取含中文的路径**（本项目路径含"中国软件杯"）。用 `np.frombuffer` + `cv2.imdecode` 绕过。

**⑤ 硬链接跨卷失败**：

- **现象 A**：`os.link` 跨分区抛 `OSError`
- **现象 B**：`dataN/val` 为空，YOLO 报 "No images found in val"
- **修复**：统一封装 `hardlink_or_copy(src, dst)`，`except (OSError, NotImplementedError)` 时回退 `shutil.copy2`
- ⚠️ **唯一例外**：`step8/tool/init.py` 的 `data1/images` 用的是 `copy2`（这里正是"硬链接导致 val 为空"的现场）

---

## 五、⚠️ 环境/资源类陷阱

| 陷阱 | 现象 | 规避 |
|:---|:---|:---|
| **僵尸 python 进程累积** | 物理内存 28/32 GB，训练从 0.3 s/batch 恶化到 **30 s/batch**（100 倍） | 定期 `taskkill` 清理残留 `python.exe`，检查 `nvidia-smi` |
| **训练极慢** | 第一个 epoch 高达 80 s/batch | 确保独享 GPU |
| **OOM** | CUDA memory allocation failure（Step8 每个 epoch 都触发） | 清理 GPU 后重试，或降 batch（**但会静默丢参数**，见下） |
| **Resume 失败** | 恢复 checkpoint 时 CUDA OOM | **不 resume**，直接用 `last.pt` 作为 `weights` 参数重新训练 |

**⚠️ 降 batch 的隐性代价（Step8 全轮踩中）**：`step8/tool/train.py` 的 OOM 重试分支**漏传 `iou`** 且把 `epochs` 重置为 `HP["epochs"]`，导致全部 15 轮的 `--iou 0.9` 从未生效（所有 train 日志均为 `Batch used: 8`）。

> **教训**：**异常/重试分支必须传齐所有参数**。否则参数会**静默回退默认值**，产生无法从日志发现的错误结论。

---

## 六、延伸阅读

- 转换环境为何必须匹配 Paddle 版本 → [02-X2Paddle补丁记录.md](02-X2Paddle补丁记录.md)
- 全部 27 条踩坑 → [../lessons/27条踩坑与修复全记录.md](../lessons/27条踩坑与修复全记录.md)
- Step8 因 OOM 分支漏参导致的结论失效 → [../docs/09-B榜-Step8-多框感知三阶段自举.md](../docs/09-B榜-Step8-多框感知三阶段自举.md)
