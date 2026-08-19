# 数据集迁移方案：wangjia184 数据集 → bchao1/Anime-Face-Dataset

> 整理日期：2026-08-19
> 适用版本：基于 V2.0 teacher（FID 32.17）之后的下一次训练（下文称 V3.0）
> 结论速览：**模型本体一行不用改，必改项只有 1 处代码（尺寸校验）+ 4 处配置；另有一批强烈建议与顺手改进项。**

---

## 一、旧数据集与旧模型的诊断（为什么要换）

### 1.1 旧数据集实际情况（datasets/splits_v1_2/metadata.json）

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| 原始图片 | 21,418 张 | 来源 wangjia184/diffusion_model |
| 精确重复（SHA-256） | 4,473 张（21%） | 源数据质量一般 |
| 近似重复（pHash≤4） | 939 张 | 去重管线已剔除 |
| 去重后总量 | 16,006 张 | 真实可学分布 |
| train / val / test | 14,406 / 800 / 800 | 5% 测试集只有 800 张 |
| 分辨率 | 64×64（固定） | 旧数据的信息量上限 |

### 1.2 核心问题清单

**P0-1 数据规模瓶颈（最严重）**
14,406 张训练图喂约 50M 参数的 V2.0 U-Net（约 0.3 张图/参数；作为对照，ADM 在 ImageNet 上约 2.3 张图/参数）。V2.0 参数量约为 V1.2 的 2.5 倍、加了注意力，FID 却只从 36.47 降到 32.17（11.8%）——模型容量已跑在数据前面，继续扩容模型收益递减。

**P0-2 评估协议失真（误导后续所有决策）**
evaluate.py 的 FID 计算中，reference 侧和生成侧都只有 800 张（test 集大小）：

- 800 样本的 FID 系统性偏高、方差大，真实 FID 可能明显低于 32.17；
- V1.2 → V2.0 的 11.8% 改善可能部分落在小样本噪声范围内（KID 方向一致支持真实改善，但置信度打折）；
- 任何小改动（loss 加权、t 采样分布等）在 800 样本协议下大概率测不出来。

**P1-1 训练量不足**
14,406 × 120 epochs ÷ 32 ≈ 54k 步，对 50M 参数的流模型偏少（同类工作常见 200k–500k+ 步）。当前是调度器决定的"结束"，不是模型潜力的结束。

**P1-2 记忆化未验证**
小数据 + 大模型最容易记住训练样本，FID/KID 测不出，需要最近邻分析（生成样本 vs 训练集的 L2 距离分布）。

**P2 技术层面小问题（便宜但值得顺手做）**

| 问题 | 位置 | 改法 |
| --- | --- | --- |
| t 均匀采样，端点信噪比极端 | flow/rectified_flow.py | `time = torch.sigmoid(torch.randn(...))`（logit-normal，SD3 验证过） |
| 正弦嵌入按 t∈[0,1000] 设计的频率喂 t∈[0,1]，近半维度无区分度 | models/time_embedding.py | `angles = (time * 1000.0)[:, None] * frequencies[None, :]` |
| attention 只在 16/8，32×32 层缺全局建模 | config | `attention_resolutions: [32, 16, 8]`（纯配置） |
| pHash 阈值 4 偏松，仍有同尺寸残留近重复 | config | `phash_threshold: 3` 并抽查 dedup_report.json |

**结论：换数据集是当前收益最大的单项改动，其余都是配套。**

---

## 二、新数据集情况（bchao1/Anime-Face-Dataset）

### 2.1 基本信息

| 项目 | 内容 |
| --- | --- |
| 数量 | 63,632 张（约为旧数据集去重后的 **4 倍**） |
| 来源 | www.getchu.com 爬取，经 lbpcascade_animeface（nagadomi）人脸检测裁剪 |
| 分辨率 | **不固定，90×90 ~ 120×120**，含少量非正方形图 |
| 质量 | 作者宣称"high quality"：干净背景、色彩丰富，优于杂乱的 Danbooru 系 |
| 已知问题 | 少量坏裁剪、少量非人脸离群图 |
| 格式 | Kaggle 打包版为 PNG（数字命名，`images/` 目录）；GitHub 原始抓取脚本产物为 JPG（`src/cropped/`） |
| 获取 | 具体数据集已放入目录：Data/train/64k |


### 2.2 与旧数据集对比

| 维度 | 旧（wangjia184） | 新（bchao1） |
| --- | --- | --- |
| 去重后规模 | 16,006 | 预计 60k 上下（待 pHash 去重确认） |
| 分辨率 | 64×64 固定 | 90–120px 可变（下采样到 64 只损失少量高频） |
| 源质量 | 21% 精确重复 | 高质量、干净背景 |
| 风格分布 | 来源混杂 | getchu 游戏立绘脸，分布更集中 |

### 2.3 迁移必须知道的三件事

1. **FID 与 V2.0 的 32.17 不可直接比较**——参考分布换了。必须在新数据集上重建基线（如需对照，可把 V2.0 权重在新 test 集上重跑一次 evaluate）。
2. 尺寸不固定这件事**只卡数据准备脚本**（校验函数），不卡模型——dataset.py 的 `transforms.Resize((64, 64))` 本来就会把任意输入缩到 64×64。
3. 训练集变成约 57,000 张（63,632 × 90%），同样 epochs 下迭代步数约 ×4，注意训练时长与磁盘占用。

---

## 三、新模型需要修改的地方

### 3.1 必改项（不改会报错或产出错误结果）

**① `datasets/split_dataset.py` — `validate_image()` 尺寸校验（唯一必改代码）**

现状：要求图片严格等于 `(expected_size, expected_size)`，新数据集 90–120px 可变尺寸会**全军覆没**。

```python
# 修改前
if image.size != (expected_size, expected_size):
    raise ValueError(f"图片尺寸不是 {expected_size}x{expected_size}: {path} ({image.size})")

# 修改后：只要求短边不小于目标尺寸（90px > 64px，全部合格），非正方形图交给加载时的缩放/裁剪
if min(image.size) < expected_size:
    raise ValueError(f"图片短边小于 {expected_size}px: {path} ({image.size})")
```

顺带删除 `image.mode != "RGB"` 的强校验（P/RGBA 模式图片让 dataset.py 已有的 `convert("RGB")` 处理即可），否则调色板模式图片会被误杀。

**② `datasets/split_dataset.py` — 文件扫描只认 `.png`**

```python
# 修改前
image_paths = sorted(raw_dir.glob("*.png"), key=lambda path: path.name)

# 修改后：兼容 png/jpg/jpeg（Kaggle 版是 PNG，GitHub 脚本产物是 JPG）
image_paths = sorted(
    (p for p in raw_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}),
    key=lambda path: path.name,
)
```

**③ `config/v3_0_bchao.yaml`（新建）— 数据与输出路径全部换新**

```yaml
data:
  raw_dir: Data/anime_face_bchao/images   # 指向实际图片目录
  split_dir: datasets/splits_v3_0_bchao   # 新清单目录，绝不复用 splits_v1_2
  auto_prepare: true                       # 或手动跑 --force

paths:
  output_dir: outputs/v3_0_bchao
  checkpoint_dir: outputs/v3_0_bchao/checkpoints
  sample_dir: outputs/v3_0_bchao/samples
  plot_dir: outputs/v3_0_bchao/plots
  evaluation_dir: outputs/v3_0_bchao/evaluation
  log_dir: runs/rectified_flow_v3_0_bchao
```

路径隔离的目的：`resume: true` 会读取 checkpoint_dir 下的 `latest.pt`，目录为空则自动从头训练，不会误把 V2.0 权重续训到新数据上（数据分布已变，即使结构兼容也不应续用）。

**④ 重新生成数据清单**

```bash
python -m datasets.split_dataset --config config/v3_0_bchao.yaml --force
```

生成后检查 `metadata.json` 的去重统计，抽查 `dedup_report.json` 确认没有大面积误删。

### 3.2 强烈建议项（不阻塞，但直接影响效果与结论可信度）

| # | 改动 | 位置 | 理由 |
| --- | --- | --- | --- |
| ⑤ | 裁剪顺序改为 `Resize(64) + CenterCrop(64)`（短边缩放后中心裁剪），替换 `Resize((64,64))` 直接压扁 | datasets/dataset.py | 新数据集有非正方形图，直接拉伸会轻微变形；近正方形图两者几乎无差别，但改了更稳 |
| ⑥ | `phash_threshold: 4 → 3`，并抽查 dedup_report.json | config | 63k 数据里同角色/同 CG 的近似帧更多；同时上次发现旧清单仍有同尺寸残留近重复 |
| ⑦ | `epochs: 120 → 60~80`（首轮） | config | 训练图 ×4，60 epochs ≈ 10.7 万步已超 V2.0 总步数两倍；先看曲线再决定是否加训 |
| ⑧ | `save_every_epochs: 10 → 20` | config | 单个 checkpoint 约 800MB+（模型+EMA+优化器），120 epochs × 10 会吃 ~10GB 磁盘 |
| ⑨ | **评估协议修复**（趁重建基线一起做）：reference 改用新数据集全量（约 60k），生成侧固定 ≥10,000 张；加最近邻记忆化检查 | evaluate.py | 不修的话新基线同样是 800 样本的失真值，后续所有对照又会失效 |
| ⑩ | 有效 batch 16×2=32 → 32×2=64 或更高（看显存） | config | 数据 ×4 后梯度更稳，大 batch 通常带来 FID 收益 |

### 3.3 零改动项（架构已参数化，自动适配）

| 环节 | 为什么不用改 |
| --- | --- |
| models/unet.py、blocks.py、attention.py | `image_size` 是构造参数；保持 `image_size: 64` 则网络结构、attention 分辨率校验（16/8）完全不变 |
| flow/rectified_flow.py | 训练目标与图像尺寸无关 |
| train.py / sample.py / evaluate.py | 噪声形状、采样、FID 张量流全部读 `config["data"]["image_size"]` |
| datasets/dataset.py 的归一化 | `Normalize(±1)` 对任意输入成立 |

### 3.4 顺手一起做的模型改进（与换数据解耦，可各出一版对照）

这些是第一部分诊断出的 P2 项，改动极小，建议在 V3.0 直接带上（若要严格归因，可用消融开关）：

1. t 采样改 logit-normal（flow/rectified_flow.py，一行）；
2. 时间嵌入频率补偿 ×1000（models/time_embedding.py，一行）；
3. `attention_resolutions: [16, 8] → [32, 16, 8]`（config，一行）。

### 3.5 不建议做的事

- **换数据与改分辨率同时进行**：变量太多，无法归因。先固定 64×64 验证数据收益；新数据 90–120px 的分辨率红利留给 V3.1（`image_size: 96` 时需同步改 `attention_resolutions: [24, 12]`，否则 unet.py 的层级分辨率校验会直接抛 ValueError）。
- **复用/续训 V2.0 权重**：数据分布已变，从头训。
- **立刻混入旧 16k 数据**：两套来源风格分布不同，先纯新数据建立基线，之后可作为消融实验再试混合。

---

## 四、执行步骤

1. 从 Kaggle 下载 `splcher/animefacedataset`（GitHub 原仓库已私有），解压到 `Data/anime_face_bchao/`，确认 `raw_dir` 指向含 63,632 张图的目录；
2. 按 3.1 修改 `validate_image()` 与文件扫描后缀；
3. 新建 `config/v3_0_bchao.yaml`（新 split_dir / paths / epochs / phash_threshold，带上 3.4 的三项小改进）；
4. `python -m datasets.split_dataset --config config/v3_0_bchao.yaml --force`，核对去重统计与离群图；
5. （建议）修改 evaluate.py 评估协议：reference 全量 + 生成 10k；
6. `python train.py --config config/v3_0_bchao.yaml`；
7. 训练完成后 `python evaluate.py --config config/v3_0_bchao.yaml`，建立新基线；
8. （可选）把 V2.0 的 best.pt 在新 test 集上重跑一次 evaluate，得到同协议下的新旧对照。

## 五、验收标准

- 数据清单生成成功，去重后规模在 55k–63k 区间，无大面积误删；
- 训练曲线正常收敛（train/val loss 同步下降，无发散）；
- 新基线 FID（全量 reference + 10k 生成）显著低于 V2.0 在同一新协议下的重测值——这才是"数据扩充有效"的干净证据；
- 最近邻检查无严重记忆化迹象。

---

*本方案基于 V2.0 代码（commit c9e3262）与 bchao1/Anime-Face-Dataset 的公开说明整理。*
