# test_image_file.py
import cv2
import numpy as np
import argparse
from success_predictor import SuccessPredictor
from utils import load_config

def load_and_preprocess_image(image_path: str, target_size: int) -> np.ndarray:
    """严格匹配原系统要求的图像加载函数"""
    # 读取图像
    img = cv2.imread(image_path)  # 返回BGR格式的numpy数组
    if img is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
    
    # 转换为RGB格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 确保尺寸匹配配置要求
    if img_rgb.shape[:2] != (target_size, target_size):
        img_resized = cv2.resize(img_rgb, (target_size, target_size))
    else:
        img_resized = img_rgb
        raise ValueError(f"格式错误: {image_path}")
    
    return img_resized.astype(np.uint8)

def main():
    # 配置参数解析
    parser = argparse.ArgumentParser(description='测试任务成功预测器')
    parser.add_argument('--image_path', type=str, required=True, 
                       help='测试图像路径（支持jpg/png）')
    parser.add_argument('--task', type=str, required=True,
                       help='用英文描述的任务指令（需与训练prompt格式一致）')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径（默认：config.yaml）')
    args = parser.parse_args()

    # 加载配置文件
    config = load_config(args.config)
    predictor = SuccessPredictor(config)
    
    # 加载并预处理图像
    target_size = config["general_params"]["shoulder_camera_image_size"]
    test_image = load_and_preprocess_image(args.image_path, target_size)
    
    # 执行预测
    success = predictor.predict_outcome(
        image=test_image,
        task_str=args.task,
        log_metrics=True
    )
    
    # 输出结果
    print("\n" + "="*50)
    print(f"Task: {args.task}")
    print(f"Image: {args.image_path}")
    print(f"Success Prediction: {'TRUE' if success else 'FALSE'}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()