import braille_inference_picture as vision
import cv2


def print_braille_results(braille_results):
    """打印转换结果"""
    for i, line in enumerate(braille_results, 1):
        print(f"第{i}行: ", end="")
        for j, braille_set in enumerate(line):
            print(f"{braille_set}", end="")
            if j < len(line) - 1:
                print(" | ", end="")
        print()

def display_braille_grid(braille_set):
    """以3x2网格形式显示盲文点"""
    grid = [['○', '○'],
            ['○', '○'], 
            ['○', '○']]
    
    # 盲文点编号对应网格位置
    point_positions = {
        1: (0, 0), 2: (1, 0), 3: (2, 0),
        4: (0, 1), 5: (1, 1), 6: (2, 1)
    }
    
    for point in braille_set:
        if point in point_positions:
            row, col = point_positions[point]
            grid[row][col] = '●'
    
    for row in grid:
        print(' '.join(row))

def capture_and_recognize_braille(show_image=False):
    """
    捕获摄像头图像，进行盲文识别，并返回结果。
    
    Args:
        show_image (bool): 是否显示标注后的结果图片。
        
    Returns:
        list: 识别出的盲文点阵集合，例如 [[{1, 2}, {3, 4}]]。
              如果失败则返回空列表。
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return []

    try:
        ret, frame = cap.read()
        if not ret:
            print("无法从摄像头捕获图像")
            return []

        # 运行推理
        annotated_image_path, detections_txt_path, braille_sets = vision.braille_inference_run(True, frame)
        
        # 打印结果路径
        print("\n--- 视觉识别结果 ---")
        print(f"标注图片: {annotated_image_path}")
        print(f"检测文本: {detections_txt_path}")
        
        # 打印识别出的盲文点阵
        print("识别出的盲文点阵:")
        print_braille_results(braille_sets)

        if show_image:
            annotated_image = cv2.imread(annotated_image_path)
            if annotated_image is not None:
                cv2.imshow("Braille Recognition Result", annotated_image)
                cv2.waitKey()
                cv2.destroyAllWindows()
            else:
                print(f"无法读取结果图片: {annotated_image_path}")
        
        return braille_sets

    except Exception as e:
        print(f"在捕获和识别过程中发生错误: {e}")
        return []
    finally:
        cap.release()
        print("摄像头已释放")


if __name__ == "__main__":
    print("正在执行独立的视觉识别测试...")
    # 调用新封装的函数进行测试，并显示结果图片
    recognized_braille = capture_and_recognize_braille(show_image=True)
    
    if recognized_braille:
        print("\n测试成功，函数返回的盲文点阵为:")
        print(recognized_braille)
    else:
        print("\n测试失败或未识别到任何内容。")