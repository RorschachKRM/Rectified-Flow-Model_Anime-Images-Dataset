# 基于动漫头像的 Rectified Flow（V3.0）实现文档

新数据来源：
https://github.com/bchao1/Anime-Generation.git


https://github.com/bchao1/Anime-Face-Dataset.git

[V1.0 - V2.0实现文档](/V1.0%20-%20V2.0模型实现文档.md)

---

## 1. V3.0目标与版本边界

V3.0的目标是在新的大规模数据集上训练一个稳定的**无条件64×64动漫头像Rectified Flow基础模型**。本版本只接收噪声和时间步，不使用发色、瞳色或年份标签。

版本规划如下：

- V3.0：使用清洗后的72k图片训练无条件基础模型；
- V4.0：加载V3.0基础权重，在35k发色/瞳色标注图片上进行条件微调；
- V3.0与V4.0的数据清单、checkpoint、日志和评估结果全部使用独立目录。

V3.0继续使用V2.0已经验证过的62.19M参数U-Net，首轮只改变数据规模、数据预处理和评估协议，不同时加入新的时间采样、时间嵌入或32×32 Attention改动，以便明确判断数据扩充的实际收益。

## 2. 数据集选择

### 2.1 数据来源

V3.0使用两个原始图片目录：

```text
Data/train/64k/
Data/train/anime_generation_color/images/
```

第一部分来自Anime-Face-Dataset，文件名带有年份后缀，例如`10000_2004.jpg`。第二部分来自Anime-Generation中的发色/瞳色标注数据，文件名为纯数字，例如`10000.jpg`。


经检查，该Anime-Generation下载包实际由两部分拼接而成：

- 63,569张带年份图片，基本与项目中已有的`Data/train/64k`相同；
- 36,740张纯数字64×64图片，这是相对于已有数据真正新增的部分。

因此没有再次复制63k年份数据，只将36,740张新增图片移动到项目中，避免保存两套重复源文件。

### 2.2 数据移动结果

新增图片和V4.0需要的标签已移动到：

```text
Data/train/anime_generation_color/
├── images/             # 36,740张纯数字JPG图片
├── tags.csv            # 每张图片的发色、瞳色标签
├── hair_label.json     # 12种发色到类别编号的映射
└── eye_label.json      # 10种瞳色到类别编号的映射
```


### 2.3 选择这两部分数据的原因

年份数据的优势是部分源图片高于64×64，整体边缘和细节比新增颜色数据更清晰；新增颜色数据的优势是精确重复率较低、人物和颜色分布更丰富，并且所有图片已经是64×64。

新增颜色数据存在一定柔化和画风同质性，单独使用可能让结果偏软。因此V3.0将两部分混合：年份数据提供相对清晰的结构，颜色数据扩充分布和样本量。两部分精确哈希没有交集，不是同一批文件的重复副本。

## 3. 数据质量检查与清洗规则

### 3.1 原始数据统计

| 项目 | 数量 |
| --- | ---: |
| 扫描到的JPG图片 | 100,305 |
| 年份来源图片 | 63,565 |
| 新增颜色来源图片 | 36,740 |
| 无法读取的损坏图片 | 0 |
| 短边低于64px而拒绝的图片 | 6,329 |
| 通过尺寸和完整性校验的文件 | 93,976 |
| 精确重复副本 | 21,247 |
| 最终唯一可用图片 | 72,729 |

### 3.2 清洗规则

V3.0采用以下规则：

1. 同时扫描`.jpg`、`.jpeg`和`.png`；
2. 使用Pillow验证每个图片文件是否可读取；
3. 只保留短边不小于64px的图片；
4. 读取时统一执行`convert("RGB")`，不因P、RGBA等模式直接拒绝图片；
5. 对所有来源统一计算SHA-256，执行全局精确去重；
6. 先去重、后划分，确保重复图不会跨越train/val/test；
7. V3.0关闭pHash近似去重。

不使用现有pHash近似去重的原因是：动漫人脸构图高度相似，当前“原图+中心裁剪变体取最小哈希距离”的实现曾把不同人物错误合并。精确去重结果可靠，因此V3.0只使用SHA-256去重。

所有拒绝图片、精确重复簇和处理参数记录在：

```text
datasets/splits_v3_0_unconditional/dedup_report.json
datasets/splits_v3_0_unconditional/metadata.json
```

### 3.3 数据划分

V3.0使用固定随机种子42，按85%/5%/10%划分：

| Split | 总数 | 年份来源 | 颜色来源 |
| --- | ---: | ---: | ---: |
| train | 61,820 | 31,704 | 30,116 |
| val | 3,636 | 1,865 | 1,771 |
| test | 7,273 | 3,668 | 3,605 |
| 合计 | 72,729 | 37,237 | 35,492 |

验收结果：

- 三个split中所有路径均存在；
- 每个split内部路径和SHA-256均唯一；
- train/val/test之间路径交集为0；
- train/val/test之间SHA-256交集为0；
- 清单中图片的最小短边为64px。

生成的清单位于：

```text
datasets/splits_v3_0_unconditional/
├── train.txt
├── val.txt
├── test.txt
├── metadata.json
└── dedup_report.json
```

## 4. V3.0代码修改

### 4.1 多数据源配置

`utils/config.py`增加了`data.raw_dirs`支持，同时保留旧版本的`data.raw_dir`兼容性。配置加载时会将所有数据路径解析成项目绝对路径，并检查：

- `raw_dirs`必须是非空列表；
- 不能重复配置相同数据目录；
- `min_source_size`必须大于0；
- 图片扩展名必须以`.`开头；
- 评估真实集合只能是train、val或test；
- `num_generated`必须大于0。

### 4.2 数据准备

`datasets/split_dataset.py`完成了以下改造：

- 支持一个或多个原始图片目录；
- 支持JPG/JPEG/PNG混合数据；
- 将严格64×64校验改为可配置的最小源图短边；
- 无法读取和尺寸不足图片不再中断整个流程，而是跳过并记录原因；
- 在所有来源之间统一执行SHA-256去重；
- inventory使用项目相对路径，避免不同目录中同名文件互相覆盖；
- metadata、三份manifest和dedup report全部存在时才允许使用缓存；
- 路径处理不再为10万文件重复调用文件系统级`resolve()`，首次强制准备约86秒，未变化时缓存检查约5秒。

### 4.3 图像变换

`datasets/dataset.py`将原来的强制拉伸：

```python
transforms.Resize((64, 64))
```

改为：

```python
transforms.Resize(64, antialias=True)
transforms.CenterCrop(64)
```

这样会先按比例缩放短边，再中心裁剪，避免非正方形图片被拉伸。训练集继续使用随机水平翻转，所有图片最终归一化到`[-1, 1]`。

### 4.4 评估协议

`evaluate.py`将真实图片数量和生成图片数量解耦。旧实现会按照test大小生成相同数量的图片，V1/V2因此只有800个FID样本。V3.0配置为：

```yaml
evaluation:
  real_split: test
  num_generated: 10000
  kid_subset_size: 1000
```

Flow MSE始终使用test；FID/KID使用`real_split`指定的真实数据，并独立生成10,000张图片。评估结果分别记录`num_real_samples`和`num_generated_samples`。

## 5. V3.0模型与训练配置

完整配置文件：`config/v3_unconditional.yaml`。

### 5.1 模型

| 参数 | 数值 |
| --- | --- |
| 图像尺寸 | 64×64 |
| 输入/输出通道 | 3/3 |
| base channels | 96 |
| channel multipliers | [1, 2, 4, 4] |
| 每层残差块 | 2 |
| 时间嵌入维度 | 384 |
| dropout | 0.05 |
| Attention分辨率 | [16, 8] |
| Attention heads | 8 |
| 总参数量 | 62,188,515 |

V3.0没有条件Embedding、类别标签、文本编码器或Cross-Attention，是纯无条件生成模型。

### 5.2 训练

| 参数 | 数值 |
| --- | --- |
| epochs | 60 |
| 微批次 | 16 |
| 梯度累积 | 2 |
| 有效batch size | 32 |
| 初始学习率 | 2e-4 |
| 调度器 | CosineAnnealingLR |
| 最低学习率 | 1e-6 |
| AdamW weight decay | 1e-4 |
| EMA decay | 0.9999 |
| 混合精度 | 开启 |
| 梯度裁剪 | 1.0 |
| 验证间隔 | 每1 epoch |
| 预览间隔 | 每5 epochs |
| 固定快照间隔 | 每20 epochs |

61,820张训练图训练60 epochs约为116k次optimizer更新，是V2.0约54k次更新的两倍以上。

### 5.3 采样

```yaml
sampling:
  solver: heun
  num_steps: 50
  num_samples: 16
  trajectory_frames: 9
  trajectory_samples: 8
```

## 6. 运行方法

### 6.1 环境

```powershell
conda activate code
```

### 6.2 重新生成数据清单

清单已经生成。只有在修改原始数据或清洗配置后才需要强制重建：

```powershell
python -m datasets.split_dataset --config config/v3_unconditional.yaml --force
```

正常训练配置中`auto_prepare: true`，文件没有变化时会直接复用缓存。

### 6.3 开始或恢复训练

```powershell
python train.py --config config/v3_unconditional.yaml
```

V3.0使用独立的checkpoint目录。首次运行时没有`latest.pt`，因此从头训练；训练中断后再次执行相同命令，会从V3.0自己的下一epoch恢复。

### 6.4 TensorBoard

```powershell
tensorboard --logdir runs/rectified_flow_v3_0_unconditional
```

### 6.5 独立采样

```powershell
python sample.py --config config/v3_unconditional.yaml --seed 123
```

### 6.6 评估

训练得到`best.pt`后运行：

```powershell
python evaluate.py --config config/v3_unconditional.yaml
```

评估默认使用EMA权重、Heun 50步、7,273张test真实图片和10,000张生成图片。

## 7. 输出目录

```text
outputs/v3_0_unconditional/
├── checkpoints/
│   ├── latest.pt
│   ├── best.pt
│   ├── epoch_0020.pt
│   ├── epoch_0040.pt
│   └── epoch_0060.pt
├── samples/
├── plots/
└── evaluation/

runs/rectified_flow_v3_0_unconditional/
```

单个V3.0完整checkpoint预计约949MB。`latest.pt`和`best.pt`会覆盖更新，固定epoch快照每20 epochs保存一次。

## 8. 实现验收

V3.0实现完成后执行了以下检查：

- 23项Pytest测试全部通过；
- V3.0配置路径和V2.0输出路径完全隔离；
- 多数据源扫描、尺寸拒绝、RGB转换和跨目录精确去重测试通过；
- 评估独立生成数量测试通过；
- 真实DataLoader可以读取`[batch, 3, 64, 64]`张量；
- 输入像素范围为`[-1, 1]`；
- 62.19M参数模型可以正常完成Rectified Flow前向loss计算；
- 完整数据清单通过路径、尺寸和跨split哈希泄漏检查。

## 9. 与V2.0指标的关系

V2.0的FID 32.17来自旧数据分布和800张test图片。V3.0更换了真实数据分布，并采用7,273张test真实图片和10,000张生成图片，因此V3.0 FID不能与V2.0的32.17直接比较。

V3.0训练完成后应将其评估结果作为新协议基线。后续V3.x或V4.0只有在复用同一test清单、EMA、采样器、步数和随机种子时，FID/KID才可以直接比较。

## 10. V4.0预留

V3.0不会读取`tags.csv`、`hair_label.json`或`eye_label.json`。V4.0计划使用35,492张精确去重后的颜色标注图片，在V3.0无条件基础权重上加入发色和瞳色Embedding，并使用Classifier-Free Guidance进行条件微调。

由于年份数据没有发色/瞳色标签，V4.0条件微调只使用纯数字颜色数据；V3.0预训练则让条件模型在微调前已经学习到完整72k分布的人脸结构和风格。
