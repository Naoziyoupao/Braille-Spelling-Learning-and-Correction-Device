import cv2
import torch
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import argparse
import time
import itertools

class YOLOInference:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        初始化YOLO推理类
        
        Args:
            model_path: 训练好的模型路径 (.pt文件)
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_names = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """加载训练好的YOLO模型"""
        try:
            print(f"加载模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # 获取类别名称
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                print(f"类别名称: {self.class_names}")
            else:
                print("警告: 无法获取类别名称")
            
            print(f"使用设备: {self.device}")
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise
    
    def preprocess_image(self,is_img, image_path):
        """
        预处理图像
        Args:
            is_img: 是否为图像False为图像路径True为图像
            image_path: 输入图像路径
            
        Returns:
            image: 处理后的图像
            original_image: 原始图像
        """
        # 读取图像
        if is_img:
            image = image_path
        else:
            image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        #TODO 对拍照的图像的预处理比如裁剪
        original_image = image.copy()
        
        print(f"图像尺寸: {image.shape}")
        return image, original_image
    
    def inference(self, image):
        """
        对单张图像进行推理
        
        Args:
            image: 输入图像
            
        Returns:
            results: 推理结果
        """
        # 使用模型进行推理
        results = self.model(
            image, 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False  # 不输出详细信息
        )
        
        return results
    
    def calculate_iou(self, box1, box2):
        """
        计算两个边界框的IoU（交并比）
        
        Args:
            box1: 第一个边界框 [x1, y1, x2, y2]
            box2: 第二个边界框 [x1, y1, x2, y2]
            
        Returns:
            iou: IoU值
        """
        # 计算交集区域
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        # 计算交集面积
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
        # 计算两个框的面积
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 计算并集面积
        union_area = box1_area + box2_area - inter_area
        
        # 计算IoU
        iou = inter_area / union_area if union_area > 0 else 0
        
        return iou
    
    def remove_overlapping_boxes(self, detections_info, iou_threshold=0.5, method='confidence'):
        """
        去除重叠的检测框（对不同类别的框也进行去重）
        
        Args:
            detections_info: 检测信息列表
            iou_threshold: IoU阈值，超过此阈值认为框重叠
            method: 去重方法，可选 'confidence'（保留置信度高的）或 'smaller'（保留面积小的）
            
        Returns:
            filtered_detections: 过滤后的检测信息
        """
        if not detections_info:
            return detections_info
        
        # 按置信度降序排序
        sorted_detections = sorted(detections_info, key=lambda x: x['confidence'], reverse=True)
        filtered_detections = []
        
        while sorted_detections:
            # 取出置信度最高的检测
            current = sorted_detections.pop(0)
            filtered_detections.append(current)
            
            # 找出与当前检测框不重叠的检测（不考虑类别）
            non_overlapping = []
            for detection in sorted_detections:
                iou = self.calculate_iou(current['bbox'], detection['bbox'])
                
                # 如果IoU小于阈值，保留该检测
                if iou < iou_threshold:
                    non_overlapping.append(detection)
                else:
                    # 重叠框的处理
                    if method == 'smaller':
                        # 保留面积较小的框
                        current_area = current['bbox_size'][0] * current['bbox_size'][1]
                        detection_area = detection['bbox_size'][0] * detection['bbox_size'][1]
                        if detection_area < current_area:
                            # 用面积小的替换当前框
                            filtered_detections[-1] = detection
                    # 默认方法：保留置信度高的（已经在前面排序处理了）
            
            sorted_detections = non_overlapping
        
        print(f"去重叠前: {len(detections_info)} 个检测框")
        print(f"去重叠后: {len(filtered_detections)} 个检测框")
        print(f"移除了 {len(detections_info) - len(filtered_detections)} 个重叠框")
        
        # 统计移除的类别信息
        removed_classes = {}
        original_classes = {}
        for det in detections_info:
            class_name = det['class_name']
            original_classes[class_name] = original_classes.get(class_name, 0) + 1
        
        for det in filtered_detections:
            class_name = det['class_name']
            removed_classes[class_name] = original_classes.get(class_name, 0) - 1
        
        print("各类别移除情况:")
        for class_name, count in removed_classes.items():
            if count > 0:
                print(f"  {class_name}: 移除了 {count} 个")
        
        return filtered_detections
    
    def draw_detections(self, image, results, show_conf=True, show_labels=True, remove_overlaps=False, iou_threshold=0.5, remove_method='confidence'):
        """
        在图像上绘制检测结果
        
        Args:
            image: 原始图像
            results: 推理结果
            show_conf: 是否显示置信度
            show_labels: 是否显示标签
            remove_overlaps: 是否去除重叠框
            iou_threshold: 去重叠的IoU阈值
            remove_method: 去重叠方法
            
        Returns:
            annotated_image: 标注后的图像
            detections_info: 检测信息列表
        """
        annotated_image = image.copy()
        detections_info = []
        
        # 遍历所有检测结果
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 获取置信度和类别
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    # 获取类别名称
                    class_name = self.class_names[cls_id] if self.class_names else str(cls_id)
                    
                    # 存储检测信息
                    detections_info.append({
                        'class_id': cls_id,
                        'class_name': class_name,
                        'confidence': float(conf),
                        'bbox': [x1, y1, x2, y2],
                        'bbox_center': [(x1+x2)//2, (y1+y2)//2],
                        'bbox_size': [x2-x1, y2-y1]
                    })
        
        # 去除重叠框（如果启用）- 现在对所有类别进行去重
        if remove_overlaps:
            detections_info = self.remove_overlapping_boxes(
                detections_info, iou_threshold, remove_method
            )
        
        # 绘制过滤后的检测框
        for detection in detections_info:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            conf = detection['confidence']
            cls_id = detection['class_id']
            
            # 绘制边界框
            color = self.get_color(cls_id)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # 准备标签文本
            label = ""
            if show_labels:
                label += f"{class_name}"
            if show_conf:
                label += f" {conf:.2f}"
            
            # 绘制标签背景
            if label:
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_image, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            color, -1)
                
                # 绘制标签文本
                cv2.putText(annotated_image, label, 
                          (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_image, detections_info
    
    def get_color(self, class_id):
        """根据类别ID生成颜色"""
        # 使用固定的颜色映射
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 黄色
            (128, 0, 128),  # 深紫色
            (0, 128, 128),  # 橄榄色
            (128, 128, 0),  # 深青色
            (128, 0, 0)     # 深蓝色
        ]
        return colors[class_id % len(colors)]
    
    def print_detection_summary(self, detections_info):
        """打印检测结果摘要"""
        print("\n=== 检测结果摘要 ===")
        print(f"检测到的目标数量: {len(detections_info)}")
        
        # 按类别统计
        class_counts = {}
        for detection in detections_info:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} 个")
        
        # 显示置信度统计
        if detections_info:
            confidences = [d['confidence'] for d in detections_info]
            print(f"平均置信度: {np.mean(confidences):.3f}")
            print(f"最高置信度: {np.max(confidences):.3f}")
            print(f"最低置信度: {np.min(confidences):.3f}")
    
    def save_results(self,is_img, image_path, annotated_image, detections_info, output_dir="./inference_results"):
        """
        保存推理结果, 使用固定文件名以覆盖旧文件。
        
        Args:
            is_img:是图像还是图像路径
            image_path: 原始图像路径
            annotated_image: 标注后的图像
            detections_info: 检测信息
            output_dir: 输出目录
        """
        # 创建输出目录
        Path(output_dir).mkdir(exist_ok=True)
        
        # 使用固定的文件名，而不是时间戳
        output_image_path = Path(output_dir) / "latest_result.jpg"
        output_txt_path = Path(output_dir) / "latest_detections.txt"
        
        # 保存标注图像
        cv2.imwrite(str(output_image_path), annotated_image)
        print(f"标注图像已保存: {output_image_path}")
        
        # 定义行高阈值，用于判断是否在同一行（可根据需要调整）
        row_threshold = 50  # 像素值，可根据实际情况调整

        # 按Y坐标分组，将Y坐标相近的检测框视为同一行
        rows = {}
        for detection in detections_info:
            y_center = detection['bbox_center'][1]
            # 查找是否已经有相近Y坐标的行
            found_row = False
            for row_y in rows.keys():
                if abs(y_center - row_y) <= row_threshold:
                    rows[row_y].append(detection)
                    found_row = True
                    break
            # 如果没有找到相近的行，创建新行
            if not found_row:
                rows[y_center] = [detection]
        # 按行排序（从上到下）
        sorted_rows = sorted(rows.items(), key=lambda item: item[0])

        # 对每行内的检测框按X坐标排序（从左到右）
        sorted_detections = []
        for row_y, row_detections in sorted_rows:
            sorted_row = sorted(row_detections, key=lambda d: d['bbox_center'][0])
            sorted_detections.extend(sorted_row)

        with open(output_txt_path, 'w', encoding='utf-8') as f:
            # 按行输出，同一行的类别在同一行用空格分隔
            for row_y, row_detections in sorted_rows:
                # 对当前行按X坐标排序
                sorted_row = sorted(row_detections, key=lambda d: d['bbox_center'][0])
                # 提取类别名称
                class_names = [detection['class_name'] for detection in sorted_row]
                # 写入一行，类别用空格分隔
                f.write(" ".join(class_names) + "\n")
        
        print(f"检测详情已保存: {output_txt_path}")
        
        return str(output_image_path), str(output_txt_path)

# 简化使用函数（如果不使用命令行参数）
def simple_inference(model_path, is_img ,image_path, output_dir="./inference_results", conf_threshold=0.25,iou_threshold = 0.3, remove_overlaps=False, overlap_iou=0.5, remove_method='confidence'):
    """
    简化的推理函数
    
    Args:
        model_path: 模型路径
        image_path: 图像路径
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        iou_threshold: IOU阈值
        remove_overlaps: 是否去除重叠框
        overlap_iou: 重叠框IoU阈值
        remove_method: 去重叠方法
        
    Returns:
        annotated_image_path: 标注图像路径
        detections_info: 检测信息
    """
    inference_engine = YOLOInference(model_path, conf_threshold=conf_threshold, iou_threshold = iou_threshold)
    # 预处理
    image, original_image = inference_engine.preprocess_image(is_img,image_path)
    # 推理
    results = inference_engine.inference(image)
    # 绘制结果
    annotated_image, detections_info = inference_engine.draw_detections(
        original_image, results, 
        remove_overlaps=remove_overlaps,
        iou_threshold=overlap_iou,
        remove_method=remove_method
    )
    # 保存结果
    annotated_image_path, annotated_txt_path = inference_engine.save_results(
        is_img, image_path, annotated_image, detections_info, output_dir
    )
    
    # 打印摘要
    inference_engine.print_detection_summary(detections_info)
    
    return annotated_image_path, annotated_txt_path, detections_info
"""""""""""""""""""""""""""""""""""""""上面是模型推理部分，下面是盲文转换部分和文件接口"""""""""""""""""""""""""""""""""""""""""""""
# 定义编号到盲文点组合的映射
def create_braille_mapping():
    points = [1, 2, 3, 4, 5, 6]
    all_combinations = []
    for k in range(1, 7):
        for comb in itertools.combinations(points, k):
            all_combinations.append(set(comb))
    
    mapping = {i+1: all_combinations[i] for i in range(len(all_combinations))}
    return mapping

# 创建映射表
braille_mapping = create_braille_mapping()

def number_to_braille_points(num):
    """将编号转换为盲文点集合"""
    return braille_mapping.get(num, set())

def read_and_convert_braille(file_path):
    """从文件读取编号并转换为盲文集合"""
    braille_results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                # 分割每行的数字
                numbers = line.split()
                line_braille = []
                
                for num_str in numbers:
                    try:
                        num = int(num_str)
                        if 1 <= num <= 63:
                            braille_set = number_to_braille_points(num)
                            line_braille.append(braille_set)
                        else:
                            print(f"警告: 第{line_num}行数字 {num} 超出范围(1-63)，忽略")
                            line_braille.append(set())
                    except ValueError:
                        print(f"警告: 第{line_num}行包含无效数字 '{num_str}'，忽略")
                        line_braille.append(set())
                
                braille_results.append(line_braille)
                
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return []
    except Exception as e:
        print(f"错误: 读取文件时发生异常 - {e}")
        return []
    
    return braille_results



"""文件对外的接口函数"""
def braille_inference_run(is_img, img):
    print("braille_inference_run")
    annotated_image_path, output_path, detections = simple_inference(
        "/home/raspbarrypi/Desktop/dachuang/vision/best.pt", 
        is_img,
        img, 
        conf_threshold=0.4, 
        iou_threshold=0.2,
        remove_overlaps=True,      # 启用重叠框去除
        overlap_iou=0.3,           # 设置IoU阈值
        remove_method='confidence' # 使用置信度优先的方法
    )
    braille_sets = read_and_convert_braille(output_path)
    return annotated_image_path, output_path, braille_sets

if __name__ == "__main__":
    
    # 直接调用
    model_path = "/home/raspbarrypi/Desktop/dachuang/vision/best.pt"
    image_path = "/home/raspbarrypi/Desktop/dachuang/vision/real_test_picture/1.jpg"
    
    # 启用重叠框去除（现在会对所有类别进行去重）
    annotated_image_path, output_path, detections = simple_inference(
        model_path, 
        False,
        image_path, 
        conf_threshold=0.4, 
        iou_threshold=0.2,
        remove_overlaps=True,      # 启用重叠框去除
        overlap_iou=0.3,           # 设置IoU阈值
        remove_method='confidence' # 使用置信度优先的方法
    )
    print(f"结果保存至: {output_path}")