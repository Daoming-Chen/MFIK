# MeanFlow IK - Robot Inverse Kinematics using MeanFlow

## 概述

本项目实现了基于MeanFlow生成模型的机器人逆运动学（IK）求解系统。MeanFlow是一种高效的生成模型框架，通过Neighborhood Projection方法学习从末端位姿到关节配置的映射。

## 安装

确保已安装所有依赖：

```bash
pip install -r mfik/requirements.txt
```

## 快速开始

### 1. 生成数据集

从URDF文件生成训练数据集。系统会随机采样关节配置，并通过正向运动学计算对应的末端位姿：

```bash
python -m mfik.dataset_generator \
    --urdf robots/panda_arm.urdf \
    --num-samples 100000 \
    --output data/panda_dataset.pt \
    --end-effector panda_link8
```

**参数说明：**
- `--urdf`: 机器人URDF文件路径（必需）
- `--num-samples`: 生成的样本数量（默认100000）
- `--output`: 输出数据集路径，保存为`.pt`格式（必需）
- `--end-effector`: 末端执行器link名称（必需）
- `--seed`: 随机种子，用于可重复性（默认42）

数据集会自动划分为训练集(70%)、验证集(15%)和测试集(15%)。

### 2. 训练模型

使用生成的数据集训练MeanFlow IK模型。推荐使用Neighborhood Projection和课程学习：

```bash
python -m mfik.train \
    --data data/panda_dataset.pt \
    --output-dir checkpoints/panda \
    --batch-size 256 \
    --epochs 100 \
    --lr 1e-4 \
    --noise-std-min 0.1 \
    --noise-std-max 1.0 \
    --curriculum
```

**核心参数：**
- `--data`: 数据集路径（必需）
- `--output-dir`: checkpoint保存目录（必需）
- `--batch-size`: 批次大小（默认256）
- `--lr`: 学习率（默认1e-4）
- `--epochs`: 训练轮数（默认100）
- `--save-interval`: checkpoint保存间隔（默认10）
- `--device`: 计算设备（cuda/cpu，自动检测）
- `--resume`: 从checkpoint恢复训练的路径

**Neighborhood Projection参数：**
- `--noise-std-min`: 生成参考配置q_ref的最小噪声标准差（默认0.1）
- `--noise-std-max`: 生成参考配置q_ref的最大噪声标准差（默认1.0）
- `--curriculum`: 启用课程学习，噪声从大到小逐渐减小（推荐使用）

**MeanFlow高级参数：**
- `--time-mu`: Logit-normal时间采样均值（默认-0.4）
- `--time-sigma`: Logit-normal时间采样标准差（默认1.0）
- `--adaptive-p`: 自适应权重参数（默认0.0）

训练过程中会在`output-dir/logs`目录下生成TensorBoard日志，可通过以下命令查看：

```bash
tensorboard --logdir checkpoints/panda/logs
```

### 3. 评估模型

评估训练好的模型在测试集上的性能：

```bash
python -m mfik.evaluate \
    --checkpoint checkpoints/panda/final_model.pt \
    --data data/panda_dataset.pt \
    --urdf robots/panda_arm.urdf \
    --end-effector panda_link8 \
    --num-steps 1 \
    --output evaluation_report.json
```

**参数说明：**
- `--checkpoint`: 模型checkpoint路径（必需）
- `--data`: 测试数据集路径（必需）
- `--urdf`: URDF文件路径（必需）
- `--end-effector`: 末端执行器link名称（必需）
- `--num-steps`: 采样步数（默认1，单步生成，推荐用于快速推理）
- `--output`: 评估报告保存路径（默认`evaluation_report.json`）
- `--device`: 计算设备（cuda/cpu，自动检测）
- `--batch-size`: 评估批次大小（默认256）
- `--use-gt-ref`: 使用真实关节作为q_ref（轨迹跟踪模式）

评估结果包括：
- **位置误差**（Position Error）：末端位置的欧几里得距离误差
- **旋转误差**（Rotation Error）：末端旋转的角度误差
- **关节违规率**（Joint Violation Rate）：超出关节限制的样本比例
- **推理速度**（Inference Time）：每个样本的平均推理时间

## 推理模式

MeanFlow IK支持两种推理模式：

1. **全局求解模式**（默认）：使用随机q_ref，可以发现所有可能的IK解
2. **轨迹跟踪模式**（`--use-gt-ref`）：使用当前关节配置作为q_ref，找到最接近当前状态的IK解

## 示例工作流

完整的工作流示例：

```bash
# 1. 生成Panda机器人数据集
python -m mfik.dataset_generator \
    --urdf robots/panda_arm.urdf \
    --num-samples 100000 \
    --output data/panda_dataset.pt \
    --end-effector panda_link8

# 2. 训练模型
python -m mfik.train \
    --data data/panda_dataset.pt \
    --output-dir checkpoints/panda \
    --batch-size 8192 \
    --epochs 400 \
    --lr 1e-4 \
    --noise-std-min 0.1 \
    --noise-std-max 1.0 \
    --curriculum \
    --amp

# 3. 评估模型
python -m mfik.evaluate \
    --checkpoint checkpoints/panda_run1/final_model.pt \
    --data data/panda_dataset.pt \
    --urdf robots/panda_arm.urdf \
    --end-effector panda_link8 \
    --output evaluation_panda.json
```

## 性能调优建议

- **数据集规模**：建议至少10万样本，更多样本可提升泛化能力
- **批次大小**：GPU内存允许的情况下使用更大的batch size（256-512）
- **课程学习**：启用`--curriculum`可以提升训练稳定性和最终性能
- **噪声范围**：根据机器人的关节范围调整`--noise-std-min`和`--noise-std-max`
- **多步采样**：评估时可尝试`--num-steps 5`或更多，可能提升精度但会降低速度

