import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import glob
import os
import sys
from PIL import Image, ImageDraw, ImageFont

# 将llava.py所在目录添加到Python路径
llava_path = os.path.dirname(os.path.abspath(__file__))
llava_path = os.path.join(llava_path, "..", "..")  # 向上两级到项目根目录
sys.path.insert(0, llava_path)

# 导入llava的describe_bbox_target函数
try:
    from llava import describe_bbox_target
    LLAVA_AVAILABLE = True
    print("TNL2KDataset: LLaVA模块可用")
except ImportError as e:
    LLAVA_AVAILABLE = False
    print(f"TNL2KDataset: LLaVA模块不可用: {e}")


class TNL2KDataset(BaseDataset):
    """
    TNL2K dataset - 模板+标注版本
    在图像上绘制边界框，然后调用LLaVA的describe_bbox_target函数
    """
    def __init__(self, use_llava_description=True):
        """
        初始化TNL2K数据集
        Args:
            use_llava_description: 是否使用LLaVA生成描述（True=使用，False=使用原始描述）
        """
        super().__init__()
        self.base_path = self.env_settings.tnl2k_path
        self.use_llava_description = use_llava_description
        
        # 创建保存标注图像的目录
        self.save_dir = os.path.join(self.base_path, "llava_annotated_images")
        os.makedirs(self.save_dir, exist_ok=True)

    def get_sequence_list(self):
        sequences = self._get_sequence_list()
        return SequenceList([self._construct_sequence(s) for s in sequences if s is not None])

    def _draw_bbox_on_image(self, image_path, bbox):
        """在图像上绘制边界框（与OTBDataset相同）"""
        try:
            # 打开图像
            image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            # 绘制红色边界框
            x, y, w, h = bbox
            draw.rectangle([x, y, x + w, y + h], outline='red', width=3)
            
            # 添加文字标签
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            draw.text((x + 5, y + 5), "目标", fill='red', font=font)
            
            return image
            
        except Exception as e:
            print(f"[TNL2KDataset] 绘制边界框失败: {e}")
            return Image.open(image_path).convert('RGB')

    def _construct_sequence(self, sequence_name):
        """构建序列 - 模板+标注版本"""
        # 构建路径
        anno_path = os.path.join(self.base_path, sequence_name, 'groundtruth.txt')
        frames_dir = os.path.join(self.base_path, sequence_name, 'imgs')
        
        # 检查文件是否存在
        if not os.path.exists(anno_path):
            print(f"警告: {sequence_name} 的groundtruth.txt不存在")
            return None
            
        if not os.path.exists(frames_dir):
            print(f"警告: {sequence_name} 的imgs目录不存在")
            return None
        
        # 加载ground truth数据
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        
        # 构建帧列表
        frames_list = sorted(glob.glob(os.path.join(frames_dir, '*')))
        
        if len(frames_list) == 0:
            print(f"警告: 序列 {sequence_name} 没有图像文件")
            return None

        # 语言描述文件路径
        # 原始语言文件路径
        original_language_file = os.path.join(self.base_path, sequence_name, "language.txt")
        # LLaVA生成的语言文件路径（保存在原始路径）
        language_file_llava = os.path.join(self.base_path, sequence_name, "language_llava.txt")
        
        # 语言描述
        language = ""
        
        # 检查是否已有LLaVA生成的描述
        if os.path.exists(language_file_llava):
            with open(language_file_llava, 'r', encoding='utf-8') as f:
                language = f.readlines()[0].rstrip()
                print(f"[TNL2KDataset] 使用已保存的LLaVA描述: {sequence_name}")
        elif self.use_llava_description and LLAVA_AVAILABLE:
            try:
                print(f"\n[TNL2KDataset] 处理序列: {sequence_name}")
                
                # 第一帧路径
                first_frame_path = frames_list[0]
                
                if os.path.exists(first_frame_path):
                    # 1. 获取第一帧的边界框
                    first_bbox = ground_truth_rect[0] if len(ground_truth_rect) > 0 else ground_truth_rect[0]
                    
                    # 验证边界框
                    x, y, w, h = first_bbox
                    if w > 0 and h > 0:
                        # 2. 在图像上绘制边界框
                        annotated_image = self._draw_bbox_on_image(first_frame_path, first_bbox)
                        
                        # 3. 保存带标注的图像
                        seq_save_dir = os.path.join(self.save_dir, sequence_name)
                        os.makedirs(seq_save_dir, exist_ok=True)
                        
                        # 获取第一帧文件名
                        first_frame_name = os.path.basename(first_frame_path)
                        annotated_path = os.path.join(seq_save_dir, f"annotated_{first_frame_name}")
                        annotated_image.save(annotated_path, 'JPEG')
                        print(f"  标注图像已保存: {annotated_path}")
                        
                        # 4. 使用带标注的图像调用LLaVA的describe_bbox_target
                        print(f"  使用LLaVA分析带标注的目标...")
                        language = describe_bbox_target(annotated_path, first_bbox, simple_output=True)
                        
                        # 5. 提取描述内容
                        if language.startswith("DESCRIPTION:"):
                            language = language[len("DESCRIPTION:"):].strip()
                        print(f"  LLaVA生成描述: {language}")
                        
                        # 6. 保存语言描述到文件（language_llava.txt）
                        os.makedirs(os.path.dirname(language_file_llava), exist_ok=True)
                        with open(language_file_llava, 'w', encoding='utf-8') as f:
                            f.write(language)
                    else:
                        print(f"  - 警告: 边界框尺寸为0，使用原始描述")
                        language = self._get_original_language(original_language_file)
                else:
                    print(f"  - 警告: 第一帧不存在，使用原始描述")
                    language = self._get_original_language(original_language_file)
                    
            except Exception as e:
                print(f"  - LLaVA处理失败: {e}")
                language = self._get_original_language(original_language_file)
        else:
            # 使用原始描述
            language = self._get_original_language(original_language_file)
        
        print("========================")
        print(language)
        print("========================")
        
        # 返回序列
        return Sequence(
            sequence_name,
            frames_list,
            'tnl2k',
            ground_truth_rect.reshape(-1, 4),  # 原始边界框数据，保持不变
            object_class=None,
            target_visible=None,
            language=language
        )

    def _get_original_language(self, language_file_path):
        """获取原始的语言描述"""
        if os.path.exists(language_file_path):
            try:
                with open(language_file_path, 'r', encoding='utf-8') as f:
                    language = f.readlines()[0].rstrip()
                    return language
            except:
                return ""
        else:
            return ""

    def __len__(self):
        sequences = self._get_sequence_list()
        return len([s for s in sequences if s is not None])

    @property
    def sequence_list(self):
        """获取序列列表"""
        return self._get_sequence_list()

    def _get_sequence_list(self):
        sequence_list = sorted([p.split('/')[-2] for p in glob.glob(os.path.join(self.base_path, '*/'))])
        return sequence_list
