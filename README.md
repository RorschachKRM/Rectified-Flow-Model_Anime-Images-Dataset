# 基于动漫头像的 Rectified Flow（V1.0）

数据来源：https://github.com/wangjia184/diffusion_model.git


这是一个面向深度学习初学者的无条件图片生成练习项目。项目使用 PyTorch、残差 U-Net 和最基础的 Rectified Flow，不包含 reflow，也不使用类别或文本条件。

项目支持使用 Conda 管理本地 Python 环境。代码、数据、模型权重和 TensorBoard 日志均保存在当前项目目录。

## 1. 模型原理

定义：

```text
x0 ~ N(0, I)                    # 源分布：高斯噪声
x1 ~ p_data                     # 目标分布：真实图片
t  ~ Uniform(0, 1)
xt = (1 - t) * x0 + t * x1     # 线性插值
目标速度 = x1 - x0
```

U-Net 接收 `xt` 和时间 `t`，学习速度场：

```text
v_theta(xt, t) ≈ x1 - x0
loss = MSE(v_theta(xt, t), x1 - x0)
```

生成时从新的高斯噪声出发，用前向欧拉法从 `t=0` 积分到 `t=1`：

```text
x <- x + dt * v_theta(x, t)
```

## 2. U-Net 网络设计

模型不是直接预测最终图片，而是接收插值状态 `x_t` 和时间 `t`，输出与图片形状相同的速度场 `v_theta(x_t, t)`。默认输入输出均为 `[B, 3, 64, 64]`，可训练参数约为 23.72M。

默认配置：

```yaml
model:
  in_channels: 3
  out_channels: 3
  base_channels: 64
  channel_multipliers: [1, 2, 4, 4]
  num_res_blocks: 2
  time_embedding_dim: 256
  dropout: 0.0
```

整体结构：

```text
x_t: [B, 3, 64, 64]
        │
        ▼
3×3 Conv → [B, 64, 64, 64]
        │
        ├── Encoder level 0: 2×ResBlock, 64 ch, 64×64
        │                    ↓ 3×3 stride-2 Conv
        ├── Encoder level 1: 2×ResBlock, 128 ch, 32×32
        │                    ↓ 3×3 stride-2 Conv
        ├── Encoder level 2: 2×ResBlock, 256 ch, 16×16
        │                    ↓ 3×3 stride-2 Conv
        └── Encoder level 3: 2×ResBlock, 256 ch, 8×8
                             │
                             ▼
                  Middle: 2×ResBlock, 256 ch
                             │
                             ▼
        ┌── Decoder level 3: 3×ResBlock, 256 ch, 8×8
        │                    ↑ nearest interpolation + 3×3 Conv
        ├── Decoder level 2: 3×ResBlock, 256 ch, 16×16
        │                    ↑ nearest interpolation + 3×3 Conv
        ├── Decoder level 1: 3×ResBlock, 128 ch, 32×32
        │                    ↑ nearest interpolation + 3×3 Conv
        └── Decoder level 0: 3×ResBlock, 64 ch, 64×64
                             │
                             ▼
              GroupNorm → SiLU → 3×3 Conv
                             │
                             ▼
                 v_theta: [B, 3, 64, 64]
```

编码器每个分辨率层包含 2 个残差块，并在层与层之间使用 `3×3、stride=2` 卷积下采样。解码器从编码器保存的特征中按相反顺序取出 skip connection，与当前特征在通道维拼接；每个分辨率层使用 3 个残差块，以消费对应层的跳跃特征。上采样采用最近邻插值，再接一个 `3×3` 卷积。

单个带时间条件的残差块结构为：

```text
主分支：
x → GroupNorm → SiLU → 3×3 Conv
  → 加入 Linear(SiLU(time_embedding))[:, :, None, None]
  → GroupNorm → SiLU → Dropout → 3×3 Conv

捷径分支：
x → Identity                         （输入输出通道相同）
x → 1×1 Conv                         （输入输出通道不同）

输出 = 主分支 + 捷径分支
```

`GroupNorm` 最多使用 32 组；若通道数不能被 32 整除，会自动减少组数。默认 `dropout=0.0`，因此 Dropout 层不会丢弃特征。

时间 `t` 先经过 64 维正弦/余弦位置编码，然后由两层 MLP 映射为 256 维：

```text
t → SinusoidalEmbedding(64)
  → Linear(64, 256)
  → SiLU
  → Linear(256, 256)
```

同一个时间嵌入会通过各残差块独立的线性层投影到相应通道数，并以逐通道偏置的方式加入卷积特征。这样网络能够根据当前时间位置预测不同的速度场。

最后一个 `3×3` 输出卷积使用全零权重和偏置初始化，使模型训练开始时预测接近零速度。当前 V1.0 没有使用 self-attention、类别条件或文本条件，重点保持网络简单并验证 Rectified Flow 的基本流程。

## 3. 项目结构

```text
.
├── .vscode/                # VS Code 任务和调试配置
├── config/default.yaml      # 数据、模型、训练和采样配置
├── datasets/
│   ├── dataset.py           # Dataset 和 DataLoader
│   └── split_dataset.py     # 图片校验、哈希去重和数据划分
├── flow/rectified_flow.py   # 训练目标和欧拉求解器
├── models/
│   ├── blocks.py            # 残差块、上下采样
│   ├── time_embedding.py    # 正弦时间编码
│   └── unet.py              # 残差 U-Net
├── utils/                   # 配置、设备、权重和可视化工具
├── tests/                   # 小型单元测试
├── train.py                 # 训练入口
├── sample.py                # 独立生成入口
├── evaluate.py              # 测试集 loss
├── environment.yml          # Conda 环境说明
└── requirements.txt
```

## 4. 配置 Conda 环境

建议安装：

- Python：3.12
- 与设备和 CUDA 版本兼容的 PyTorch、torchvision
- VS Code Python 扩展

根据 `environment.yml` 创建环境：

```powershell
conda env create -f environment.yml
conda activate code
```

若环境已经存在，可补齐项目依赖：

```powershell
python -m pip install -r requirements.txt
```

PyTorch 和 torchvision 需要根据操作系统、显卡及 CUDA 环境单独安装。`requirements.txt` 不会覆盖现有的 PyTorch 安装。

打开项目后，按 `Ctrl+Shift+P`，执行 `Python: Select Interpreter`，选择刚创建的 Conda 环境。

激活环境后可以检查运行状态：

```powershell
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

也可以从 `终端 → 运行任务` 执行“检查 GPU 环境”“运行单元测试”“训练 Rectified Flow”和“启动 TensorBoard”。

## 5. 数据准备

默认图片目录为：

```text
Data/train/nolabel
```

项目不会移动、删除或改写原始图片。数据准备脚本会：

1. 扫描实际存在的 PNG，不依赖连续文件编号；
2. 检查图片是否可读取、是否为 RGB 64×64；
3. 用 SHA-256 识别完全相同的图片；
4. 默认逻辑去重；
5. 用固定种子按 90%/5%/5% 生成训练、验证和测试清单。

手动生成清单：

```bash
python -m datasets.split_dataset --config config/default.yaml
```

清单保存在 `datasets/splits/`。训练时若清单不存在，会自动准备。要在配置发生变化后重新划分：

```bash
python -m datasets.split_dataset --config config/default.yaml --force
```

## 6. 训练

所有主要参数都位于 `config/default.yaml`。默认 batch size 为 16；显存不足时可以调低该值。

开始训练：

```bash
python train.py --config config/default.yaml
```

训练产物：

```text
outputs/checkpoints/latest.pt       # 每个 epoch 更新，可自动恢复
outputs/checkpoints/best.pt         # 验证 loss 最低的模型
outputs/checkpoints/epoch_XXXX.pt   # 定期快照
outputs/samples/epoch_XXXX.png      # 固定噪声的生成结果
outputs/plots/loss_curve.png        # train/validation loss 曲线
runs/rectified_flow/                # TensorBoard 日志
```

若 `latest.pt` 存在且配置中的 `training.resume: true`，再次运行训练命令会自动从下一个 epoch 继续。

`latest.pt` 会记录模型、优化器、AMP scaler、epoch、global step 和 loss 历史。需要续训时请保留该文件，并避免修改 U-Net 结构。

每个 checkpoint 约 272 MB。`latest.pt` 与 `best.pt` 固定约占 544 MB；默认每 10 epoch 额外保存快照，训练至 100 epoch 预计全部权重约占 3.3 GB。

显存不足时优先减小：

```yaml
training:
  batch_size: 8
```

必要时再将 `model.base_channels` 从 64 改为 48 或 32。网络结构改变后，旧 checkpoint 将不能继续加载。

## 7. TensorBoard 展示

另开一个已激活项目 Conda 环境的终端启动：

```powershell
python -m tensorboard.main --logdir runs --host 127.0.0.1 --port 6006
```

然后访问 `http://localhost:6006`。也可以从 VS Code 的“运行任务”启动 TensorBoard。

TensorBoard 中包含：

- `loss/train_step`：训练 step loss；
- `loss/train_epoch`：训练 epoch loss；
- `loss/validation_epoch`：验证 loss；
- `training/learning_rate`：学习率；
- `samples/final`：最终生成图片；
- `samples/noise_to_image_trajectory`：从纯噪声到图片的多时间步网格；
- `samples/trajectory_frames/frame_XX`：可逐帧查看的生成过程；
- `samples/noise_to_image_video`：Linux/Colab 中额外记录的动态视频。

Windows 上 TensorBoard 的 MoviePy 临时 GIF 写入存在文件锁限制，因此本地使用过程网格和逐帧图片展示完整轨迹；这不会影响训练或采样。Colab/Linux 会额外写入视频。

轨迹网格中每一行代表一个积分时刻，从第一行的纯噪声逐渐变成最后一行的生成图片。展示帧数由以下配置控制：

```yaml
sampling:
  num_steps: 100
  trajectory_frames: 9
  trajectory_samples: 8
```

## 8. 独立生成图片

训练出 `best.pt` 后执行：

```powershell
python sample.py --config config/default.yaml
```

也可以调整种子、生成数量和欧拉步数：

```powershell
python sample.py --seed 123 --num-samples 16 --num-steps 100
```

独立采样同样会向 TensorBoard 写入最终图片和生成轨迹。

## 9. 测试集评估

```powershell
python evaluate.py --config config/default.yaml
```

这里计算的是测试集 flow-matching MSE。生成模型的 loss 不能完整代表视觉质量，因此还需要结合固定噪声生成样本观察训练过程。

## 10. 单元测试

```powershell
python -m pytest -q
```

测试覆盖 U-Net 输入输出形状、时间输入校验、欧拉积分方向和数据划分计数。它们用于快速检查代码连接，不代替完整训练。

## 11. V1.0 范围

当前版本刻意保持简单：

- 无条件生成；
- 64×64 RGB；
- 残差 U-Net；
- 线性概率路径；
- MSE 速度匹配；
- 欧拉采样；
- 无 reflow；
- 无注意力和 FID。

后续确认 V1 能稳定训练并生成合理头像后，再考虑加入 EMA、attention、Heun 求解器、FID 或 reflow。


## 12. 实际生成效果

下面的示例使用训练完成后的 `best.pt`，通过 100 步欧拉法生成：

| Seed 123 | Seed 456 |
|:---:|:---:|
| ![Seed 123 生成结果](IMG/sample_seed_123.png) | ![Seed 456 生成结果](IMG/sample_seed_456.png) |

可见模型稚嫩，效果并不好。

## V1.1 更新日志

V1.1 保持原有 U-Net、Rectified Flow 训练目标和现有模型权重不变，只完成以下三项更新：

1. **加入学习率调度器**
   - 训练由固定学习率改为 `CosineAnnealingLR` 余弦退火；
   - 初始学习率仍为 `0.0002`，最低学习率为 `0.000001`；
   - checkpoint 会保存和恢复调度器状态；
   - V1.0 checkpoint 没有调度器状态时仍可正常加载。

2. **将默认采样算法替换为 Heun 法**
   - 默认采样器由一阶 Euler 改为二阶 Heun；
   - 默认采样步数由 100 调整为 50，每一步使用起点和预测终点的平均速度进行修正；
   - 原来的 Euler 实现继续保留，可通过 `--solver euler` 用于对照；
   - 已有 V1.0 `best.pt` 可以直接使用 Heun 采样，无需重新训练。

3. **修改 Batch Size**
   - 默认训练 Batch Size 由 16 调整为 32；
   - 不改变模型结构、损失函数和数据集划分。

V1.1 默认采样命令：

```powershell
python sample.py --config config/default.yaml
```

手动指定求解器和采样步数：

```powershell
python sample.py --solver heun --num-steps 50 --seed 123
python sample.py --solver euler --num-steps 100 --seed 123
```
