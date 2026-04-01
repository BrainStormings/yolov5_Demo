import os
import cv2
import numpy as np
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import yaml
from typing import List, Tuple, Dict, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class YOLOv5DatasetProcessor:
    """YOLOv5数据集处理器：将配对图像和掩码转换为YOLOv5格式"""

    def __init__(self,
                 images_dir: Union[str, Path],
                 masks_dir: Union[str, Path],
                 output_dir: Union[str, Path] = "yolov5_dataset",
                 class_names: Optional[List[str]] = None,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 img_ext: str = ".png",
                 mask_ext: str = ".png"):
        """
        初始化处理器

        Args:
            images_dir: 原始图像目录
            masks_dir: 掩码图像目录
            output_dir: 输出目录
            class_names: 类别名称列表（红外目标通常为单类别）
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            img_ext: 图像扩展名
            mask_ext: 掩码扩展名
        """
        self.images_dir = Path(images_dir).resolve()
        self.masks_dir = Path(masks_dir).resolve()
        self.output_dir = Path(output_dir).resolve()

        self.class_names = class_names or ["target"]
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self._validate_ratios()

        self.img_ext = img_ext if img_ext.startswith('.') else f'.{img_ext}'
        self.mask_ext = mask_ext if mask_ext.startswith('.') else f'.{mask_ext}'
        self.class_id_map = {name: idx for idx, name in enumerate(self.class_names)}

        self.stats = {
            'total_images': 0,
            'total_instances': 0,
            'class_distribution': {},
            'split_counts': {'train': 0, 'val': 0, 'test': 0}
        }

        self._create_directory_structure()
        self._print_init_info()

    def _print_init_info(self):
        """打印初始化信息"""
        print("初始化完成:")
        print(f"  图像目录: {self.images_dir}")
        print(f"  掩码目录: {self.masks_dir}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  类别: {self.class_names}")
        print(f"  数据集划分: 训练{self.train_ratio:.0%}/验证{self.val_ratio:.0%}/测试{self.test_ratio:.0%}")

    def _validate_ratios(self):
        """验证划分比例合理性"""
        ratios = [self.train_ratio, self.val_ratio, self.test_ratio]
        ratio_names = ['训练集', '验证集', '测试集']

        for ratio, name in zip(ratios, ratio_names):
            if not 0 <= ratio <= 1:
                raise ValueError(f"{name}比例必须在0-1之间，当前为{ratio}")

        if abs(sum(ratios) - 1.0) > 0.001:
            raise ValueError(f"划分比例总和应为1.0，当前为{sum(ratios):.3f}")

    def _create_directory_structure(self):
        """创建YOLOv5标准目录结构"""
        dirs = [
            "images/train", "images/val", "images/test",
            "labels/train", "labels/val", "labels/test"
        ]

        for dir_path in dirs:
            (self.output_dir / dir_path).mkdir(parents=True, exist_ok=True)

    def _find_paired_files(self) -> List[Tuple[Path, Path]]:
        """查找配对的图像和掩码文件"""
        # 查找图像文件
        common_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        image_files = []

        for ext in [self.img_ext] + common_exts:
            image_files = list(self.images_dir.glob(f"*{ext}"))
            if image_files:
                self.img_ext = ext
                print(f"检测到图像格式: {ext}")
                break

        if not image_files:
            raise FileNotFoundError(f"在 {self.images_dir} 中未找到图像文件")

        # 匹配对应的掩码文件
        paired_files = []
        for img_file in image_files:
            mask_candidates = [
                self.masks_dir / f"{img_file.stem}{self.mask_ext}",
                self.masks_dir / f"{img_file.name}",
                self.masks_dir / f"{img_file.stem}_mask{self.mask_ext}",
                self.masks_dir / f"{img_file.stem}_seg{self.mask_ext}",
            ]

            mask_file = next((p for p in mask_candidates if p.exists()), None)
            if mask_file:
                paired_files.append((img_file, mask_file))
            else:
                print(f"警告: 未找到 {img_file.name} 对应的掩码文件")

        if not paired_files:
            raise FileNotFoundError("未找到有效的图像-掩码文件对")

        print(f"成功匹配 {len(paired_files)} 对图像-掩码文件")
        return paired_files

    def _mask_to_bbox(self, mask_path: Path, min_area: int = 4) -> List[List[float]]:
        """
        将掩码转换为YOLOv5边界框

        Args:
            mask_path: 掩码文件路径
            min_area: 最小检测面积阈值（像素数）

        Returns:
            边界框列表，格式为 [class_id, x_center, y_center, width, height]
        """
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return []

            height, width = mask.shape
            bboxes = []

            # 二值化处理
            _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                # 转换为YOLOv5归一化坐标
                x_center = (x + w / 2.0) / width
                y_center = (y + h / 2.0) / height
                norm_w = w / width
                norm_h = h / height

                # 坐标裁剪到[0,1]范围
                x_center = np.clip(x_center, 0.0, 1.0)
                y_center = np.clip(y_center, 0.0, 1.0)
                norm_w = np.clip(norm_w, 0.0, 1.0)
                norm_h = np.clip(norm_h, 0.0, 1.0)

                bboxes.append([0, x_center, y_center, norm_w, norm_h])

                # 更新统计信息
                self.stats['total_instances'] += 1
                self.stats['class_distribution'][0] = self.stats['class_distribution'].get(0, 0) + 1

            return bboxes

        except Exception as e:
            print(f"掩码转换错误 {mask_path}: {e}")
            return []

    def _split_dataset(self, paired_files: List[Tuple[Path, Path]], seed: int = 42) -> Dict[str, List]:
        """随机划分数据集为训练/验证/测试集"""
        random.seed(seed)
        np.random.seed(seed)

        shuffled = paired_files.copy()
        random.shuffle(shuffled)

        total = len(shuffled)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        splits = {
            'train': shuffled[:train_end],
            'val': shuffled[train_end:val_end],
            'test': shuffled[val_end:]
        }

        if not splits['train']:
            raise ValueError("训练集为空，请检查数据集划分比例")

        print(f"\n数据集划分完成 (种子={seed}):")
        for name, files in splits.items():
            print(f"  {name}: {len(files)} 张图像 ({len(files) / total:.1%})")

        return splits

    def process_split(self, split_name: str, file_pairs: List[Tuple[Path, Path]]):
        """处理单个数据集划分"""
        print(f"\n处理 {split_name} 集...")

        processed = 0
        for img_file, mask_file in tqdm(file_pairs, desc=f"处理{split_name}集"):
            try:
                # 生成边界框
                bboxes = self._mask_to_bbox(mask_file)

                # 复制图像文件
                img_dest = self.output_dir / "images" / split_name / img_file.name
                shutil.copy2(img_file, img_dest)

                # 保存标签文件
                if bboxes:
                    label_path = self.output_dir / "labels" / split_name / f"{img_file.stem}.txt"
                    with open(label_path, 'w') as f:
                        for bbox in bboxes:
                            line = ' '.join(f'{x:.6f}' if isinstance(x, float) else str(x) for x in bbox)
                            f.write(line + '\n')

                processed += 1

            except Exception as e:
                print(f"处理文件 {img_file} 时出错: {e}")
                continue

        self.stats['split_counts'][split_name] = processed
        print(f"  {split_name}集完成: {processed} 张图像")

    def _save_dataset_config(self):
        """保存数据集配置文件"""
        self.stats['total_images'] = sum(self.stats['split_counts'].values())

        # YAML配置
        yaml_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': self.class_names,
            'nc': len(self.class_names)
        }

        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write("# YOLOv5红外小目标检测数据集配置\n\n")
            yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)

        # JSON详细信息
        json_config = {
            'dataset_info': {
                'name': self.output_dir.name,
                'images_dir': str(self.images_dir),
                'masks_dir': str(self.masks_dir),
                'total_images': self.stats['total_images'],
                'total_instances': self.stats['total_instances'],
                'splits': self.stats['split_counts'],
                'class_distribution': self.stats['class_distribution']
            },
            'classes': self.class_names,
            'file_extensions': {'images': self.img_ext, 'masks': self.mask_ext}
        }

        json_path = self.output_dir / "dataset_info.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_config, f, indent=2, ensure_ascii=False)

        print(f"\n配置文件已保存:")
        print(f"  YAML: {yaml_path}")
        print(f"  JSON: {json_path}")

    def _print_statistics(self):
        """打印数据集统计信息"""
        print("\n" + "=" * 50)
        print("数据集统计信息")
        print("=" * 50)

        print(f"图像总数: {self.stats['total_images']}")
        print(f"目标总数: {self.stats['total_instances']}")

        print(f"\n数据集划分:")
        for split, count in self.stats['split_counts'].items():
            if self.stats['total_images'] > 0:
                ratio = count / self.stats['total_images']
                print(f"  {split}: {count} 张图像 ({ratio:.1%})")

        if self.stats['class_distribution']:
            print(f"\n类别分布:")
            for class_id, count in self.stats['class_distribution'].items():
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"类别{class_id}"
                if self.stats['total_instances'] > 0:
                    ratio = count / self.stats['total_instances']
                    print(f"  {class_name}: {count} 个目标 ({ratio:.1%})")

        if self.stats['total_images'] > 0:
            avg = self.stats['total_instances'] / self.stats['total_images']
            print(f"\n平均每张图像的目标数: {avg:.2f}")

        print("=" * 50)

    def process_dataset(self, seed: int = 42):
        """处理整个数据集的主流程"""
        print(f"\n开始处理红外小目标数据集 (种子={seed})...")

        try:
            # 1. 查找配对文件
            paired_files = self._find_paired_files()

            # 2. 划分数据集
            splits = self._split_dataset(paired_files, seed)

            # 3. 处理各划分
            for split_name, file_pairs in splits.items():
                self.process_split(split_name, file_pairs)

            # 4. 保存配置和统计
            self._save_dataset_config()
            self._print_statistics()

            print(f"\n✅ 数据集处理完成!")
            print(f"输出目录: {self.output_dir}")

        except Exception as e:
            print(f"\n❌ 处理失败: {e}")
            raise


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(
        description='YOLOv5红外小目标数据集处理器',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 必需参数
    parser.add_argument('--images-dir', type=str, required=True,
                        help='原始图像文件夹路径')
    parser.add_argument('--masks-dir', type=str, required=True,
                        help='掩码图像文件夹路径')

    # 可选参数
    parser.add_argument('--output-dir', type=str, default='yolov5_dataset',
                        help='输出目录 (默认: yolov5_dataset)')
    parser.add_argument('--classes', type=str, nargs='+', default=['target'],
                        help='类别名称列表 (默认: ["target"])')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集比例 (默认: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='验证集比例 (默认: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='测试集比例 (默认: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("YOLOv5 红外小目标数据集处理器")
    print("=" * 60)

    try:
        processor = YOLOv5DatasetProcessor(
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            output_dir=args.output_dir,
            class_names=args.classes,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )

        processor.process_dataset(seed=args.seed)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n调试建议:")
        print("1. 检查图像和掩码目录路径")
        print("2. 确保文件名匹配")
        print("3. 检查文件格式")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 使用示例:
    # python DataLoader.py --images-dir /path/to/images --masks-dir /path/to/masks
    main()