"""
测试参数值的分布
Objectness
iou = (iou.detach() * iou_ratio + nwd.detach() * (1 - iou_ratio)).clamp(0, 1).type(tobj.dtype).

关于此处iou的转换说明
原始公式会进行值域的控制clamp(0)
而我改成clamp(0, 1)，即0~1，是由于发现了return torch.exp(-torch.sqrt(wasserstein_2) / constant)值
大部分值取不到靠近0的部分，此处进行限制小于1，避免大于1的可能

"""

import matplotlib.pyplot as plt
import numpy as np

wasserstein = np.array(0, 12.8, 0.01)
y = np.exp(-np.square(wasserstein) / 12.8)

plt.figure(figsize=(6, 6))
plt.plot(wasserstein, y)
plt.savefig("wasserstein.png")
