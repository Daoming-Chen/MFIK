#!/usr/bin/env python3
"""
MeanFlow IK Training Pipeline

直接在下方配置区域修改参数，然后运行:
    python main.py

或者使用命令行参数:
    python main.py generate   # 生成数据集
    python main.py train      # 训练模型
    python main.py evaluate   # 评估模型
    python main.py full       # 完整流程
"""
import os

# ============================================================================
# 配置区域 - 直接在这里修改参数
# ============================================================================

# 运行模式: 'generate', 'train', 'evaluate', 'full'
MODE = 'full'

# 机器人 URDF 文件路径
URDF_PATH = 'robots/panda_arm.urdf'

# 数据集配置
DATASET_PATH = 'datasets/panda_1m.pt'  # 数据集保存/加载路径
NUM_SAMPLES = 1_000_000                 # 生成的样本数量

# 模型配置
HIDDEN_DIM = 1024    # 隐藏层维度
DEPTH = 12           # 网络深度

# 训练配置
BATCH_SIZE = 2048         # 批次大小
ITERATIONS = 100_000      # 训练迭代次数
LEARNING_RATE = 2e-3      # 学习率
TRAINING_MODE = 'meanflow' # 'meanflow' 或 'baseline'

# MeanFlow 特定参数
ALPHA_MEANFLOW = 0.1   # MeanFlow 一致性权重
LAMBDA_DISP = 0.02     # 多样性损失权重

# Checkpoint 配置
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_INTERVAL = 10_000   # 保存 checkpoint 的间隔
VALIDATION_INTERVAL = 5_000    # 验证的间隔

# 评估配置
CHECKPOINT_PATH = 'checkpoints/meanflow_ik_meanflow_final.pt'  # 评估用的模型路径
EVAL_SAMPLES = 1000          # 评估样本数
VISUALIZE = True             # 是否可视化
OUTPUT_DIR = None            # 可视化输出目录 (None = 交互式显示)

# 设备配置 (None = 自动选择)
DEVICE = None

# ============================================================================
# 以下为执行代码 - 通常不需要修改
# ============================================================================

def run_generate():
    """生成 FK 数据集"""
    from mfik.dataset import generate_dataset
    
    if not os.path.exists(URDF_PATH):
        print(f"ERROR: URDF file not found: {URDF_PATH}")
        return False
    
    generate_dataset(
        urdf_path=URDF_PATH,
        output_path=DATASET_PATH,
        num_samples=NUM_SAMPLES,
        device=DEVICE
    )
    return True


def run_train():
    """训练模型"""
    from mfik.train import train_from_dataset
    
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset file not found: {DATASET_PATH}")
        return None
    
    urdf = URDF_PATH if os.path.exists(URDF_PATH) else None
    
    trainer = train_from_dataset(
        dataset_path=DATASET_PATH,
        urdf_path=urdf,
        hidden_dim=HIDDEN_DIM,
        depth=DEPTH,
        batch_size=BATCH_SIZE,
        iterations=ITERATIONS,
        lr=LEARNING_RATE,
        mode=TRAINING_MODE,
        alpha_meanflow=ALPHA_MEANFLOW,
        lambda_disp=LAMBDA_DISP,
        checkpoint_dir=CHECKPOINT_DIR,
        device=DEVICE,
        validate=True,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        validation_interval=VALIDATION_INTERVAL
    )
    return trainer


def run_evaluate():
    """评估模型"""
    from mfik.evaluate import evaluate_model
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint file not found: {CHECKPOINT_PATH}")
        return None
    if not os.path.exists(URDF_PATH):
        print(f"ERROR: URDF file not found: {URDF_PATH}")
        return None
    
    results = evaluate_model(
        checkpoint_path=CHECKPOINT_PATH,
        urdf_path=URDF_PATH,
        num_samples=EVAL_SAMPLES,
        device=DEVICE,
        visualize=VISUALIZE,
        output_dir=OUTPUT_DIR
    )
    return results


def run_full():
    """完整流程: 生成数据集 -> 训练 -> 评估"""
    print("="*80)
    print("FULL TRAINING PIPELINE")
    print("="*80)
    print(f"URDF: {URDF_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Samples: {NUM_SAMPLES:,}")
    print("="*80)
    
    # Step 1: Generate dataset
    print("\n" + "="*80)
    print("STEP 1: GENERATE DATASET")
    print("="*80)
    if not run_generate():
        return None, None
    
    # Step 2: Train model
    print("\n" + "="*80)
    print("STEP 2: TRAIN MODEL")
    print("="*80)
    trainer = run_train()
    if trainer is None:
        return None, None
    
    # Step 3: Evaluate model
    print("\n" + "="*80)
    print("STEP 3: EVALUATE MODEL")
    print("="*80)
    
    # 使用刚训练的模型进行评估
    final_checkpoint = os.path.join(CHECKPOINT_DIR, f"meanflow_ik_{TRAINING_MODE}_final.pt")
    
    from mfik.evaluate import evaluate_model
    results = evaluate_model(
        checkpoint_path=final_checkpoint,
        urdf_path=URDF_PATH,
        num_samples=EVAL_SAMPLES,
        device=DEVICE,
        visualize=VISUALIZE
    )
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Checkpoint: {final_checkpoint}")
    if results and 'success_1cm' in results:
        print(f"Final success rate (<1cm): {results['success_1cm']:.2f}%")
    
    return trainer, results


def main():
    import sys
    
    # 如果有命令行参数，使用命令行参数
    mode = MODE
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    print(f"\n{'='*80}")
    print(f"MeanFlow IK - Mode: {mode.upper()}")
    print(f"{'='*80}\n")
    
    if mode == 'generate':
        run_generate()
    elif mode == 'train':
        run_train()
    elif mode == 'evaluate':
        run_evaluate()
    elif mode == 'full':
        run_full()
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: generate, train, evaluate, full")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
