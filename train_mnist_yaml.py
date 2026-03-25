import np
import plt
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path
from torch.utils.data import DataLoader


from ultralytics.data.dataset import ClassificationDataset
from ultralytics.utils import DATASETS_DIR
from types import SimpleNamespace


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


def parse_model(yaml_path, ch=1):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    nc = data['nc']
    backbone_cfg = data['backbone']
    head_cfg = data['head']

    backbone_layers = []
    for i, (f, n, m, args) in enumerate(backbone_cfg):
        if m == 'Conv':
            c2, k, s, p = args
            layer = Conv(ch, c2, k, s, p)
            ch = c2
        elif m == 'C3':
            c2, n_bottleneck = args
            layer = C3(ch, c2, n=n_bottleneck)
            ch = c2
        elif m == 'MaxPool':
            k, s = args
            layer = nn.MaxPool2d(kernel_size=k, stride=s)
        elif m == 'AdaptiveAvgPool2d':
            output_size = args[0]
            layer = nn.AdaptiveAvgPool2d(output_size)
        else:
            raise ValueError(f"Unsupported backbone module: {m}")
        backbone_layers.append(layer)

    head_layers = []
    for i, (f, n, m, args) in enumerate(head_cfg):
        if m == 'Dropout':
            p = args[0]
            layer = nn.Dropout(p)
        elif m == 'Linear':
            out_features = args[0]
            layer = nn.Linear(ch, out_features)
            ch = out_features
        else:
            raise ValueError(f"Unsupported head module: {m}")
        head_layers.append(layer)

    class YOLOv5Classifier(nn.Module):
        def __init__(self, backbone, head):
            super(YOLOv5Classifier, self).__init__()
            self.backbone = nn.Sequential(*backbone)
            self.head = nn.Sequential(*head)
        def forward(self, x):
            x = self.backbone(x)
            x = x.view(x.size(0), -1)
            x = self.head(x)
            return x

    model = YOLOv5Classifier(backbone_layers, head_layers)
    return model, nc


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, batch in enumerate(train_loader):
        data, target = batch['img'].to(device), batch['cls'].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'====> Epoch {epoch} Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch['img'].to(device), batch['cls'].to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'====> Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy


def visualize_predictions(model, device, test_loader, num_samples=10):
    """可视化模型预测结果 (随机抽样，确保覆盖所有类别)"""
    model.eval()
    
    # 设置随机种子，确保每次运行结果不同
    np.random.seed()
    torch.manual_seed(torch.randint(0, 10000, (1,)).item())
    
    # 获取所有测试数据
    all_data = []
    all_targets = []
    
    print("正在加载测试集数据...")
    with torch.no_grad():
        for batch in test_loader:
            data = batch['img']
            target = batch['cls']
            all_data.append(data)
            all_targets.append(target)
    
    # 合并所有数据
    all_data = torch.cat(all_data, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    print(f"测试集总样本数：{len(all_data)}")
    
    # 统计各类别数量
    print("\n各类别样本数量:")
    for i in range(10):
        count = (all_targets == i).sum().item()
        print(f"数字 {i}: {count} 个样本")
    
    # 限制样本数量不超过子图数量
    max_samples = min(num_samples, len(all_data))
    
    # 随机选择索引
    total_samples = len(all_data)
    random_indices = np.random.choice(total_samples, max_samples, replace=False)
    
    # 抽取随机样本
    example_data = all_data[random_indices].to(device)
    example_target = all_targets[random_indices]
    
    print(f"\n随机抽取的样本索引：{random_indices}")
    print(f"抽取样本的真实标签：{example_target.tolist()}")
    
    # 进行预测
    with torch.no_grad():
        output = model(example_data)
        predictions = output.argmax(dim=1)
        probabilities = torch.softmax(output, dim=1)
    
    # 根据样本数量动态创建子图布局
    n_cols = 5
    n_rows = (max_samples + n_cols - 1) // n_cols  # 向上取整计算行数
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    
    # 处理 axes 的形状
    if max_samples == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(max_samples):
        img = example_data[i].cpu().numpy()
        
        # 处理图像：如果是 3 通道，转换为 HWC 格式；如果是单通道，去掉通道维
        if img.shape[0] == 3:
            # 3 通道：从 CHW 转为 HWC，并取平均转为灰度
            img = np.mean(img, axis=0)
        elif img.shape[0] == 1:
            # 单通道：直接去掉通道维
            img = img.squeeze()
        
        pred_label = predictions[i].item()
        true_label = example_target[i].item()
        prob_pred = probabilities[i, pred_label].item()
        
        # 显示图像 (使用英文标题避免字体问题)
        axes[i].imshow(img, cmap='gray')
        color = 'green' if pred_label == true_label else 'red'
        axes[i].set_title(f'Pred: {pred_label} (Conf: {prob_pred:.3f})\nTrue: {true_label}', 
                         fontsize=10, color=color)
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(max_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ 预测结果已保存到 prediction_results.png")
    plt.show()
    
    # 打印详细预测信息 (控制台仍可使用中文)
    print("\n" + "="*60)
    print("详细预测结果 (随机抽样):")
    print("="*60)
    for i in range(max_samples):
        pred_label = predictions[i].item()
        true_label = example_target[i].item()
        prob_pred = probabilities[i, pred_label].item()
        correct = "✓" if pred_label == true_label else "✗"
        print(f"样本 {i+1}: 预测={pred_label}, 真实={true_label}, 置信度={prob_pred:.4f} {correct}")
    print("="*60)
    
    # 统计预测结果
    correct_count = (predictions.cpu() == example_target.cpu()).sum().item()
    accuracy = 100. * correct_count / max_samples
    print(f"本次随机抽样准确率：{accuracy:.2f}% ({correct_count}/{max_samples})")
    
    # 计算并显示每个类别的平均置信度
    print("\n各类别预测概率分布 (前 5 个样本):")
    class_names = [str(i) for i in range(10)]
    print(f"{'样本':<8} " + " ".join([f"{name:>6}" for name in class_names]))
    print("-" * 70)
    for i in range(min(5, max_samples)):
        probs = probabilities[i].cpu().numpy()
        print(f"{i+1:<8} " + " ".join([f"{p:>6.3f}" for p in probs]))


def main():
    batch_size = 64
    epochs = 5
    learning_rate = 0.001
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")


    mnist_path = Path('D:/project/yolov5-master/mnist')
    print(f"MNIST 数据集路径：{mnist_path}")

    imgsz = 28  # MNIST 图像大小为 28x28
    
    # 创建 args 配置对象
    args = SimpleNamespace(
        imgsz=imgsz,
        fraction=1.0,
        cache=False,
        scale=0.08,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.0,
        auto_augment='randaugment',
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4
    )

    train_dataset = ClassificationDataset(str(mnist_path / 'train'), args, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


    test_dataset = ClassificationDataset(str(mnist_path / 'test'), args, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)



    yaml_file = 'D:\project\yolov5-master\models\segment\Handwritten.yaml'
    model, num_classes = parse_model(yaml_file, ch=3)
    model = model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0
    for epoch in range(1, epochs + 1):
        train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        test_acc = test(model, device, test_loader, criterion)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_mnist_model.pth')
        print()

    print(f"Best test accuracy: {best_acc:.2f}%")

    # 加载最佳模型并展示预测结果
    print("\n" + "="*60)
    print("加载最佳模型进行预测展示")
    print("="*60)
    model.load_state_dict(torch.load('best_mnist_model.pth'))
    
    # 可视化预测结果
    visualize_predictions(model, device, test_loader, num_samples=10)
    
    # 额外展示一些预测示例
    print("\n" + "="*60)
    print("第二次随机抽样预测（不同样本）:")
    print("="*60)
    model.eval()
    total_correct = 0
    total_samples = 0
    confusion_matrix = torch.zeros(10, 10, dtype=torch.long)
    
    # 再次随机抽取 15 个样本进行详细分析（增加数量以覆盖更多类别）
    visualize_predictions(model, device, test_loader, num_samples=15)
    
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch['img'].to(device), batch['cls'].to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            total_correct += pred.eq(target).sum().item()
            total_samples += target.size(0)
            
            # 构建混淆矩阵
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    overall_accuracy = 100. * total_correct / total_samples
    print(f"\n整体测试集准确率：{overall_accuracy:.2f}%")
    
    # 显示每个类别的准确率
    print("\n各类别准确率:")
    print("-" * 40)
    for i in range(10):
        class_correct = confusion_matrix[i, i].item()
        class_total = confusion_matrix[i, :].sum().item()
        class_acc = 100. * class_correct / class_total if class_total > 0 else 0
        print(f"数字 {i}: {class_correct}/{class_total} ({class_acc:.2f}%)")
    
    print("\n训练和评估完成！")

if __name__ == "__main__":
    main()