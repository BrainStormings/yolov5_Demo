#!/usr/bin/env python
# test_all_improvements.py
"""
测试所有改进功能
"""

import sys
import os
sys.path.append('.')

import torch
import yaml
from pathlib import Path


def test_lr_scheduler():
    """测试学习率调度器"""
    print("="*60)
    print("测试学习率调度器")
    print("="*60)

    from utils.improved.lr_scheduler import create_lr_scheduler

    # 创建虚拟优化器
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(2, 2))], lr=0.01)

    # 测试三种调度器
    schedulers = [
        ('cosine', 'Cosine + Warmup'),
        ('linear', 'Linear + Warmup'),
        ('step', 'Step + Warmup'),
    ]

    results = {}

    for scheduler_type, name in schedulers:
        print(f"\n测试 {name}:")

        if scheduler_type == 'cosine':
            scheduler = create_lr_scheduler(
                optimizer, scheduler_type, 100,
                warmup_epochs=5, warmup_start_lr=1e-7, eta_min=1e-6
            )
        elif scheduler_type == 'linear':
            scheduler = create_lr_scheduler(
                optimizer, scheduler_type, 100,
                warmup_epochs=5, warmup_start_lr=1e-7
            )
        elif scheduler_type == 'step':
            scheduler = create_lr_scheduler(
                optimizer, scheduler_type, 100,
                warmup_epochs=5, warmup_start_lr=1e-7,
                step_size=30, gamma=0.1
            )

        # 记录学习率变化
        lrs = []
        for epoch in range(100):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        results[name] = lrs

        print(f"  初始学习率: {lrs[0]:.2e}")
        print(f"  最大学习率: {max(lrs):.2e}")
        print(f"  最终学习率: {lrs[-1]:.2e}")
        print(f"  Warmup阶段: 0-5 epochs")

    return results


def test_iou_loss():
    """测试IoU损失函数"""
    print("\n" + "="*60)
    print("测试IoU损失函数")
    print("="*60)

    from utils.improved.loss import bbox_iou

    # 创建测试边界框
    box1 = torch.tensor([0.5, 0.5, 0.3, 0.4])  # [x_center, y_center, width, height]
    box2 = torch.tensor([0.6, 0.6, 0.4, 0.3])

    # 测试各种IoU
    iou_types = ['iou', 'giou', 'diou', 'ciou', 'eiou', 'siou']
    results = {}

    print("边界框1: [0.5, 0.5, 0.3, 0.4]")
    print("边界框2: [0.6, 0.6, 0.4, 0.3]")
    print()

    for iou_type in iou_types:
        if iou_type == 'iou':
            iou_value = bbox_iou(box1.unsqueeze(0), box2.unsqueeze(0))
        else:
            kwargs = {iou_type.upper(): True}
            iou_value = bbox_iou(box1.unsqueeze(0), box2.unsqueeze(0), **kwargs)

        results[iou_type] = iou_value.item()
        print(f"{iou_type.upper():6s}: {iou_value.item():.4f}")

    return results


def test_hyperparameter_evolution():
    """测试超参数进化"""
    print("\n" + "="*60)
    print("测试超参数进化")
    print("="*60)

    # 创建测试配置
    base_config = {
        'lr0': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,
        'warmup_start_lr': 1e-7,
        'box': 0.05,
        'cls': 0.5,
        'obj': 0.7,
        'iou_type': 'ciou',
        'optimizer': 'SGD',
        'scheduler': 'cosine',
    }

    # 保存基础配置
    os.makedirs('test_outputs', exist_ok=True)
    base_file = 'test_outputs/base_config.yaml'

    with open(base_file, 'w') as f:
        yaml.dump(base_config, f, default_flow_style=False)

    print(f"基础配置已保存到: {base_file}")
    print("\n基础配置:")
    for key, value in base_config.items():
        print(f"  {key}: {value}")

    # 模拟进化过程
    print("\n模拟进化过程...")

    # 简单变异示例
    evolved_config = base_config.copy()
    evolved_config['lr0'] = 0.01234
    evolved_config['iou_type'] = 'siou'
    evolved_config['warmup_epochs'] = 3

    # 保存进化后的配置
    evolved_file = 'test_outputs/evolved_config.yaml'
    with open(evolved_file, 'w') as f:
        yaml.dump(evolved_config, f, default_flow_style=False)

    print(f"\n进化后配置已保存到: {evolved_file}")
    print("\n进化后配置:")
    for key, value in evolved_config.items():
        if key in base_config and value != base_config[key]:
            print(f"  {key}: {value} (原始: {base_config[key]})")
        else:
            print(f"  {key}: {value}")

    return base_config, evolved_config


def test_complete_pipeline():
    """测试完整流程"""
    print("\n" + "="*60)
    print("测试完整训练流程")
    print("="*60)

    try:
        # 尝试导入训练器
        from train_final import ImprovedYOLOTrainer

        # 创建测试配置
        test_config = {
            'cfg': 'models/yolov5s.yaml',
            'nc': 80,
            'epochs': 5,  # 仅测试5个epoch
            'batch_size': 4,
            'device': 'cpu',
            'project': 'test_runs',
            'name': 'test_pipeline',
            'seed': 42,
            'exist_ok': True,
            'hyp': {
                'lr0': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 2,
                'warmup_start_lr': 1e-7,
                'box': 0.05,
                'cls': 0.5,
                'obj': 0.7,
                'iou_type': 'siou',
                'optimizer': 'SGD',
                'scheduler': 'cosine',
            }
        }

        print("创建训练器...")
        trainer = ImprovedYOLOTrainer(test_config)

        print("设置训练环境...")
        trainer.setup()

        print("构建模型...")
        trainer.build_model()

        print("设置优化器...")
        trainer.setup_optimizer()

        print("设置调度器...")
        trainer.setup_scheduler()

        print("设置损失函数...")
        trainer.setup_criterion()

        print("\n所有组件测试通过!")

        return True

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("YOLOv5改进功能测试")
    print("="*60)

    # 创建测试输出目录
    os.makedirs('test_outputs', exist_ok=True)

    # 测试1: 学习率调度器
    lr_results = test_lr_scheduler()

    # 测试2: IoU损失函数
    iou_results = test_iou_loss()

    # 测试3: 超参数进化
    base_config, evolved_config = test_hyperparameter_evolution()

    # 测试4: 完整流程
    pipeline_success = test_complete_pipeline()

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    print(f"1. 学习率调度器: 测试了 {len(lr_results)} 种调度器")
    print(f"2. IoU损失函数: 测试了 {len(iou_results)} 种IoU变体")
    print(f"3. 超参数进化: 基础配置 vs 进化配置")
    print(f"4. 完整流程测试: {'通过' if pipeline_success else '失败'}")

    print("\n所有测试输出已保存到 'test_outputs/' 目录")

    # 生成测试报告
    report_file = 'test_outputs/test_report.txt'
    with open(report_file, 'w') as f:
        f.write("YOLOv5改进功能测试报告\n")
        f.write("="*60 + "\n\n")

        f.write("1. 学习率调度器测试:\n")
        for name, lrs in lr_results.items():
            f.write(f"  {name}:\n")
            f.write(f"    初始学习率: {lrs[0]:.2e}\n")
            f.write(f"    最大学习率: {max(lrs):.2e}\n")
            f.write(f"    最终学习率: {lrs[-1]:.2e}\n\n")

        f.write("2. IoU损失函数测试:\n")
        for iou_type, value in iou_results.items():
            f.write(f"  {iou_type.upper()}: {value:.4f}\n")

        f.write("\n3. 超参数进化测试:\n")
        f.write("  基础配置已保存: test_outputs/base_config.yaml\n")
        f.write("  进化配置已保存: test_outputs/evolved_config.yaml\n")

        f.write("\n4. 完整流程测试:\n")
        f.write(f"  结果: {'通过' if pipeline_success else '失败'}\n")

    print(f"\n测试报告已保存到: {report_file}")
    print("\n测试完成!")


if __name__ == '__main__':
    main()