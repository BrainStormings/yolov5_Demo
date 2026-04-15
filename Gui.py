import json
import os
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import cv2
import torch
from PIL import Image, ImageTk


class YOLOv5_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv5 目标检测工具")
        self.root.geometry("1400x800")

        # 初始化变量
        self.model = None
        self.current_image = None
        self.current_video = None
        self.cap = None
        self.is_video_playing = False
        self.detection_results = []

        # 设置默认值
        self.conf_thres = tk.DoubleVar(value=0.25)
        self.iou_thres = tk.DoubleVar(value=0.45)
        self.selected_model = tk.StringVar()

        # 创建界面
        self.create_widgets()

        # 自动查找模型文件
        self.find_model_files()

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 模型选择部分
        model_frame = ttk.LabelFrame(control_frame, text="模型选择", padding=5)
        model_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(model_frame, text="选择模型:").pack(anchor=tk.W)
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.selected_model, state="readonly")
        self.model_combo.pack(fill=tk.X, pady=5)

        ttk.Button(model_frame, text="加载模型", command=self.load_model).pack(fill=tk.X)

        # 阈值调节部分
        threshold_frame = ttk.LabelFrame(control_frame, text="检测参数", padding=5)
        threshold_frame.pack(fill=tk.X, pady=(0, 10))

        # 置信度阈值
        ttk.Label(threshold_frame, text=f"置信度阈值: {self.conf_thres.get():.2f}").pack(anchor=tk.W)
        conf_scale = ttk.Scale(
            threshold_frame,
            from_=0.01,
            to=1.0,
            variable=self.conf_thres,
            command=lambda x: self.update_threshold_label("conf"),
        )
        conf_scale.pack(fill=tk.X, pady=5)

        # IoU阈值
        ttk.Label(threshold_frame, text=f"IoU阈值: {self.iou_thres.get():.2f}").pack(anchor=tk.W)
        iou_scale = ttk.Scale(
            threshold_frame,
            from_=0.01,
            to=1.0,
            variable=self.iou_thres,
            command=lambda x: self.update_threshold_label("iou"),
        )
        iou_scale.pack(fill=tk.X, pady=5)

        # 文件选择部分
        file_frame = ttk.LabelFrame(control_frame, text="文件操作", padding=5)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_frame, text="选择图片", command=self.select_image).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="选择视频", command=self.select_video).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="开始检测", command=self.start_detection).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="停止视频", command=self.stop_video).pack(fill=tk.X, pady=2)

        # 保存结果部分
        save_frame = ttk.LabelFrame(control_frame, text="保存结果", padding=5)
        save_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(save_frame, text="保存检测结果", command=self.save_detection_results).pack(fill=tk.X, pady=2)
        ttk.Button(save_frame, text="保存图像/视频", command=self.save_media).pack(fill=tk.X, pady=2)
        ttk.Button(save_frame, text="保存JSON结果", command=self.save_json_results).pack(fill=tk.X, pady=2)

        # 信息显示部分
        info_frame = ttk.LabelFrame(control_frame, text="检测信息", padding=5)
        info_frame.pack(fill=tk.X, expand=True)

        self.info_text = tk.Text(info_frame, height=15, width=30)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # 右侧显示区域
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 图像显示区域
        self.image_label = ttk.Label(display_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # 底部状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def find_model_files(self):
        """查找当前目录下的模型文件."""
        model_files = []
        for file in os.listdir("."):
            if file.endswith(".pt"):
                model_files.append(file)

        if model_files:
            self.model_combo["values"] = model_files
            if model_files:
                self.selected_model.set(model_files[0])
        else:
            self.selected_model.set("未找到模型文件")

    def load_model(self):
        """加载YOLOv5模型."""
        model_path = self.selected_model.get()
        if not model_path or model_path == "未找到模型文件":
            messagebox.showerror("错误", "请选择有效的模型文件")
            return

        try:
            self.status_var.set("正在加载模型...")
            # 使用torch.hub加载YOLOv5
            self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
            self.model.conf = self.conf_thres.get()
            self.model.iou = self.iou_thres.get()
            self.update_info("模型加载成功", f"模型: {model_path}\n")
            self.status_var.set("模型加载完成")
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {e!s}")
            self.status_var.set("模型加载失败")

    def update_threshold_label(self, threshold_type):
        """更新阈值显示标签."""
        if threshold_type == "conf":
            self.update_info("置信度阈值", f"已更新为: {self.conf_thres.get():.2f}")
            if self.model:
                self.model.conf = self.conf_thres.get()
        else:
            self.update_info("IoU阈值", f"已更新为: {self.iou_thres.get():.2f}")
            if self.model:
                self.model.iou = self.iou_thres.get()

    def select_image(self):
        """选择图片文件."""
        file_path = filedialog.askopenfilename(
            title="选择图片", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.current_image = file_path
            self.current_video = None
            self.stop_video()
            self.display_image(file_path)
            self.update_info("图片已选择", f"路径: {file_path}")

    def select_video(self):
        """选择视频文件."""
        file_path = filedialog.askopenfilename(
            title="选择视频", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv")]
        )
        if file_path:
            self.current_video = file_path
            self.current_image = None
            self.update_info("视频已选择", f"路径: {file_path}")
            # 显示视频第一帧
            self.display_video_preview(file_path)

    def display_image(self, image_path):
        """显示图片."""
        try:
            image = Image.open(image_path)
            # 调整图像大小以适应显示区域
            max_size = (800, 600)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # 保持引用
        except Exception as e:
            messagebox.showerror("错误", f"无法显示图片: {e!s}")

    def display_video_preview(self, video_path):
        """显示视频预览（第一帧）."""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                max_size = (800, 600)
                image.thumbnail(max_size, Image.Resampling.LANCZOS)

                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo)
                self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("错误", f"无法预览视频: {e!s}")

    def start_detection(self):
        """开始检测."""
        if not self.model:
            messagebox.showerror("错误", "请先加载模型")
            return

        if self.current_image:
            self.detect_image()
        elif self.current_video:
            self.detect_video()
        else:
            messagebox.showerror("错误", "请先选择图片或视频")

    def detect_image(self):
        """检测图片."""
        try:
            start_time = time.time()

            # 使用YOLOv5进行检测
            results = self.model(self.current_image)

            end_time = time.time()
            detection_time = (end_time - start_time) * 1000  # 转换为毫秒

            # 解析结果
            self.detection_results = []
            detections = results.pandas().xyxy[0]

            # 显示结果图像
            results_img = results.render()[0]
            results_img = cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(results_img)

            # 调整大小
            max_size = (800, 600)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            # 更新信息
            info_text = f"检测完成!\n检测时间: {detection_time:.2f}ms\n\n检测结果:\n"

            for _, row in detections.iterrows():
                result = {
                    "class": row["name"],
                    "confidence": row["confidence"],
                    "bbox": [row["xmin"], row["ymin"], row["xmax"], row["ymax"]],
                }
                self.detection_results.append(result)
                info_text += f"类别: {row['name']}, 置信度: {row['confidence']:.3f}\n"

            self.update_info("图片检测完成", info_text)
            self.status_var.set(f"检测完成 - 耗时: {detection_time:.2f}ms")

        except Exception as e:
            messagebox.showerror("错误", f"检测失败: {e!s}")

    def detect_video(self):
        """检测视频."""
        if self.is_video_playing:
            return

        def video_thread():
            try:
                self.cap = cv2.VideoCapture(self.current_video)
                self.is_video_playing = True

                fps = self.cap.get(cv2.CAP_PROP_FPS)
                frame_count = 0
                total_time = 0

                while self.is_video_playing:
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    # 检测
                    start_time = time.time()
                    results = self.model(frame)
                    end_time = time.time()
                    total_time += (end_time - start_time) * 1000

                    # 渲染结果
                    results_frame = results.render()[0]
                    results_frame = cv2.cvtColor(results_frame, cv2.COLOR_BGR2RGB)

                    # 转换为PIL图像并显示
                    image = Image.fromarray(results_frame)
                    max_size = (800, 600)
                    image.thumbnail(max_size, Image.Resampling.LANCZOS)

                    photo = ImageTk.PhotoImage(image)

                    # 在主线程中更新GUI
                    self.root.after(0, self.update_video_frame, photo)

                    # 控制帧率
                    delay = int(1000 / fps)
                    cv2.waitKey(delay)

                avg_time = total_time / frame_count if frame_count > 0 else 0
                self.update_info("视频检测完成", f"总帧数: {frame_count}\n平均检测时间: {avg_time:.2f}ms/帧")

                self.cap.release()
                self.is_video_playing = False

            except Exception:
                self.root.after(0, lambda: messagebox.showerror("错误", f"视频检测失败: {e!s}"))
                self.is_video_playing = False

        # 在新线程中运行视频检测
        thread = threading.Thread(target=video_thread)
        thread.daemon = True
        thread.start()

    def update_video_frame(self, photo):
        """更新视频帧显示."""
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def stop_video(self):
        """停止视频播放."""
        self.is_video_playing = False
        if self.cap:
            self.cap.release()
        self.status_var.set("视频已停止")

    def save_detection_results(self):
        """保存检测结果."""
        if not self.detection_results:
            messagebox.showwarning("警告", "没有检测结果可保存")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("YOLOv5 检测结果\n")
                    f.write(f"保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"置信度阈值: {self.conf_thres.get():.2f}\n")
                    f.write(f"IoU阈值: {self.iou_thres.get():.2f}\n")
                    f.write("=" * 50 + "\n\n")

                    for i, result in enumerate(self.detection_results, 1):
                        f.write(f"检测对象 {i}:\n")
                        f.write(f"  类别: {result['class']}\n")
                        f.write(f"  置信度: {result['confidence']:.3f}\n")
                        f.write(f"  边界框: {result['bbox']}\n\n")

                messagebox.showinfo("成功", f"检测结果已保存到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {e!s}")

    def save_media(self):
        """保存检测后的图像或视频."""
        if not self.detection_results and not self.is_video_playing:
            messagebox.showwarning("警告", "没有可保存的图像或视频")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("Video files", "*.mp4"), ("All files", "*.*")],
        )

        if file_path:
            try:
                # 这里需要根据实际检测结果保存图像
                # 由于显示的是调整后的图像，保存时需要重新获取原尺寸的检测结果
                messagebox.showinfo("提示", "保存功能需要根据实际图像数据进行实现")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {e!s}")

    def save_json_results(self):
        """保存JSON格式的检测结果."""
        if not self.detection_results:
            messagebox.showwarning("警告", "没有检测结果可保存")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                results_dict = {
                    "timestamp": datetime.now().isoformat(),
                    "parameters": {
                        "confidence_threshold": self.conf_thres.get(),
                        "iou_threshold": self.iou_thres.get(),
                        "model": self.selected_model.get(),
                    },
                    "detections": self.detection_results,
                }

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(results_dict, f, indent=2, ensure_ascii=False)

                messagebox.showinfo("成功", f"JSON结果已保存到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {e!s}")

    def update_info(self, title, content):
        """更新信息显示."""
        self.info_text.insert(tk.END, f"【{title}】\n{content}\n{'=' * 30}\n")
        self.info_text.see(tk.END)


def main():
    # 检查依赖
    try:
        import cv2
        import torch
        from PIL import Image, ImageTk
    except ImportError as e:
        print(f"缺少必要依赖: {e}")
        print("请安装以下包:")
        print("pip install torch torchvision opencv-python pillow")
        return

    root = tk.Tk()
    YOLOv5_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
