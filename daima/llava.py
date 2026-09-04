# test.py - Qwen3-VL模型适配器（与llava.py完全兼容）
#!/usr/bin/env python3
import torch
from PIL import Image
import sys
import os

# 尝试不同的导入方式，确保兼容性
try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    QWEN_AVAILABLE = True
except ImportError:
    try:
        # 尝试使用AutoModel作为备选
        from transformers import AutoModel, AutoProcessor
        Qwen3VLForConditionalGeneration = AutoModel  # 创建别名
        QWEN_AVAILABLE = True
        print("警告: 使用AutoModel作为Qwen3VLForConditionalGeneration的替代", file=sys.stderr)
    except ImportError:
        QWEN_AVAILABLE = False
        print("错误: 无法导入transformers模块", file=sys.stderr)

# 模型名称（使用FP8版本节省内存）
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct-FP8"

# 全局缓存变量（单例模式）- 与llava.py相同的结构
_llava_model_cache = {
    'model': None,
    'processor': None,
    'loaded': False
}

def _get_llava_model():
    """获取Qwen3-VL模型（单例模式）- 与llava.py相同的函数名"""
    if not QWEN_AVAILABLE:
        raise ImportError("Qwen3-VL模型不可用，请安装transformers>=4.40.0")
    
    if not _llava_model_cache['loaded']:
        print("Qwen3-VL: 加载模型中...", file=sys.stderr)
        try:
            _llava_model_cache['processor'] = AutoProcessor.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True
            )
            _llava_model_cache['model'] = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            _llava_model_cache['loaded'] = True
            print("Qwen3-VL: 模型加载完成", file=sys.stderr)
        except Exception as e:
            print(f"Qwen3-VL: 模型加载失败: {e}", file=sys.stderr)
            raise
    
    return _llava_model_cache['processor'], _llava_model_cache['model']

def cleanup_llava_model():
    """清理Qwen3-VL模型释放显存 - 与llava.py相同的函数名"""
    if _llava_model_cache['model'] is not None:
        del _llava_model_cache['model']
    if _llava_model_cache['processor'] is not None:
        del _llava_model_cache['processor']
    
    _llava_model_cache['model'] = None
    _llava_model_cache['processor'] = None
    _llava_model_cache['loaded'] = False
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    print("Qwen3-VL: 模型已清理", file=sys.stderr)

def describe_bbox_target(image_path, bbox, simple_output=True):
    """
    Describe the target INSIDE the given bounding box.
    与llava.py保持完全相同的函数签名和格式
    """
    import time
    start_time = time.time()
    
    if not os.path.exists(image_path):
        return "Error: Image does not exist"
    
    try:
        # 1. Load image
        image = Image.open(image_path).convert('RGB')
        img_width, img_height = image.size
        
        # 2. 使用单例模式加载模型
        processor, model = _get_llava_model()
        
        # 3. Parse bbox (ground truth from dataset)
        x, y, w, h = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        
        # 4. Enhanced prompt for detailed target description
        prompt = f"""请仔细观察图片中红色边界框内的目标对象，用一句话描述，不要太长，20个词以内：：
1. 目标是什么（人、动物、车辆、物体等具体类别）
2. 目标的颜色、形状、外观特征
3. 目标在做什么动作或处于什么状态，可选
4. 目标在图像中的具体位置（例如：图像中央、左侧、右上角等）
5. 如果有周围环境或与其他对象的关系，也请说明

请用自然流畅的英文一句话描述，只需要描述目标，不要被红色目标框影响，只描述框里面的，例如："一个穿着绿色球衣的篮球运动员，位于图像中央，正在运球，站在穿蓝色球衣球员旁边。"
"""
        
        # 5. 准备消息 (Qwen3-VL格式)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # 6. 预处理
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 7. Generate description
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # 8. 解码
        generated_ids_trimmed = generated_ids[:, inputs['input_ids'].shape[1]:]
        result = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        result = result.replace('\n', ' ').strip()
        
        # 9. 清理临时变量
        del inputs
        del generated_ids
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        total_time = time.time() - start_time
        print(f"[Qwen3-VL] Done in {total_time:.2f}s", file=sys.stderr)
        
        if simple_output:
            return f"DESCRIPTION:{result}"
        else:
            return result
        
    except Exception as e:
        print(f"Qwen3-VL: describe_bbox_target 失败: {str(e)}", file=sys.stderr)
        return f"Error: {str(e)}"

def describe_image_with_position(target_image, annotated_image, position_desc, simple_output=True):
    """
    生成目标物体的语言描述，不要太长，20个词以内
    注意：这个函数只返回文本描述，不修改边界框
    与llava.py保持完全相同的函数签名和格式
    """
    import time
    start_time = time.time()
    
    try:
        
        target_width, target_height = target_image.size
        annotated_width, annotated_height = annotated_image.size
        print(f"[Qwen3-VL] 目标特写图片像素大小: {target_width} x {target_height}", file=sys.stderr)
        print(f"[Qwen3-VL] 原始场景图片像素大小: {annotated_width} x {annotated_height}", file=sys.stderr)
        
        # 1. 获取模型（单例）
        processor, model = _get_llava_model()
        
        # 2. 优化的提示词
        prompt = f"""我提供两个图像：
第一张：被追踪物体的特写
第二张：原始场景图像

请用一句话丰富描述被追踪的物体，不要太长，20个词以内：
1. 目标是什么（人、动物、车辆、物体等具体类别）
2. 目标的颜色、形状、外观特征
3. 目标在做什么动作或处于什么状态，可选
4. 目标在图像中的具体位置（例如：图像中央、左侧、右上角等）
5. 如果有周围环境或与其他对象的关系，也请说明

请用自然流畅丰富的的英文一句话描述，只需要描述目标，例如："一个穿着绿色球衣的篮球运动员，位于图像中央，站在穿蓝色球衣球员旁边。"
此外：图像中不存在的颜色物体等不要输出，不要自己编造未知的描述，第二张原始场景图像只提供位置说明，第一张被追踪物体的特写只提供特征说明。
"""
        
        # 3. 准备消息 (Qwen3-VL格式)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": target_image},
                    {"type": "image", "image": annotated_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # 4. 预处理
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 5. 生成描述
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # 6. 解码结果
        generated_ids_trimmed = generated_ids[:, inputs['input_ids'].shape[1]:]
        result = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # 清理结果
        result = result.replace('\n', ' ').strip()
        
        # 7. 清理临时变量
        del inputs
        del generated_ids
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        
        # 8. 返回结果
        if simple_output:
            return result  # 直接返回描述，不加前缀
        else:
            return result
        
    except Exception as e:
        print(f"Qwen3-VL: 描述生成失败: {str(e)}", file=sys.stderr)
        return f"描述生成失败: {str(e)}"

def describe_image(image_path, simple_output=True):
    """
    用一句话描述图像，不要太长，30个词以内

    """
    import time
    start_time = time.time()
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        return "错误：图片不存在"
    
    try:
        # 1. 加载图片
        image = Image.open(image_path).convert('RGB')
        
        # 2. 使用单例模式加载模型
        processor, model = _get_llava_model()
        
        # 3. 固定prompt - 根据您的需求
        prompt = "请用英文一句话描述图片，不要太长，30个词以内："
        
        # 4. 准备消息 (Qwen3-VL格式)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # 5. 预处理
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 6. 生成
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # 7. 解码
        generated_ids_trimmed = generated_ids[:, inputs['input_ids'].shape[1]:]
        result = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # 清理结果
        result = result.replace('\n', ' ').strip()
        
        # 8. 清理临时变量
        del inputs
        del generated_ids
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        total_time = time.time() - start_time
        
        return result
        
    except Exception as e:
        print(f"Qwen3-VL: describe_image 错误：{str(e)}", file=sys.stderr)
        return f"错误：{str(e)}"

def main():
    """
    主函数 - 与llava.py保持完全相同的接口
    使用方法：
        python test.py [image_path] [--simple]
    """
    # 解析命令行参数
    simple_mode = False
    image_path = None
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--simple':
            simple_mode = True
        elif not arg.startswith('--'):
            image_path = arg
    
    if not image_path:
        image_path = input("请输入图片路径：").strip()
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：图片不存在 - {image_path}", file=sys.stderr)
        sys.exit(1)
    
    # 生成描述
    description = describe_image(image_path, simple_mode)
    print(description)

    # 程序结束时清理模型
    cleanup_llava_model()

if __name__ == "__main__":
    main()
