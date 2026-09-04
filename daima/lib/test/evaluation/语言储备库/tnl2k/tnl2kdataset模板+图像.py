import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import glob
import os
import sys
from PIL import Image, ImageDraw
import time

# 将llava.py所在目录添加到Python路径
llava_path = os.path.dirname(os.path.abspath(__file__))
llava_path = os.path.join(llava_path, "..", "..")  # 向上两级到项目根目录
sys.path.insert(0, llava_path)

# 导入llava函数
try:
    from llava import describe_image_with_position, cleanup_llava_model
    LLAVA_AVAILABLE = True
    print("TNL2KDataset: LLaVA模块可用")
except ImportError as e:
    LLAVA_AVAILABLE = False
    print(f"TNL2KDataset: LLaVA模块不可用: {e}")


class TNL2KDataset(BaseDataset):
    """
    TNL2K dataset - 模板+图像版本
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

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _get_position_description(self, bbox, img_width, img_height):
        """根据边界框坐标生成位置描述文本"""
        x, y, w, h = bbox
        
        # 计算中心点
        center_x = x + w/2
        center_y = y + h/2
        
        # 相对位置（0-1）
        rel_x = center_x / img_width
        rel_y = center_y / img_height
        
        # 描述水平位置
        if rel_x < 0.33:
            horiz_pos = "左侧"
        elif rel_x < 0.66:
            horiz_pos = "中间"
        else:
            horiz_pos = "右侧"
        
        # 描述垂直位置
        if rel_y < 0.33:
            vert_pos = "上方"
        elif rel_y < 0.66:
            vert_pos = "中部"
        else:
            vert_pos = "下方"
        
        # 组合描述
        if horiz_pos == "中间" and vert_pos == "中部":
            return "位于图像中央"
        else:
            return f"位于图像{horiz_pos}{vert_pos}"

    def _construct_sequence(self, sequence_name):
        """构建序列 - 模板+图像版本"""
        # 记录开始时间
        start_time = time.time()
        
        # 构建路径
        anno_path = os.path.join(self.base_path, sequence_name, 'groundtruth.txt')
        frames_dir = os.path.join(self.base_path, sequence_name, 'imgs')
        
        # 加载数据
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
        elif self.use_llava_description and LLAVA_AVAILABLE and len(frames_list) > 0:
            try:
                # 第一帧路径
                first_frame_path = frames_list[0]
                
                if os.path.exists(first_frame_path):
                    print(f"\n[TNL2KDataset] 处理序列: {sequence_name}")
                    
                    # 1. 加载原始图像
                    image = Image.open(first_frame_path).convert('RGB')
                    img_width, img_height = image.size
                    
                    # 2. 获取第一帧的边界框
                    first_bbox = ground_truth_rect[0] if len(ground_truth_rect) > 0 else ground_truth_rect[0]
                    
                    # 验证边界框
                    x, y, w, h = first_bbox
                    if w > 0 and h > 0:
                        # 确保边界框在图像范围内
                        x1 = max(0, int(x))
                        y1 = max(0, int(y))
                        x2 = min(img_width, int(x + w))
                        y2 = min(img_height, int(y + h))
                        
                        if x2 > x1 and y2 > y1:
                            # 3. 裁剪目标区域（用于给LLaVA看目标特写）
                            target_image = image.crop((x1, y1, x2, y2))
                            
                            # 4. 在原始图像上绘制边界框（仅用于可视化）
                            annotated_image = image.copy()
                            draw = ImageDraw.Draw(annotated_image)
                            draw.rectangle([x, y, x + w, y + h], outline='red', width=3)
                            
                            # 5. 生成位置描述
                            position_desc = self._get_position_description(first_bbox, img_width, img_height)
                            #print(f"  - 目标位置: {position_desc}")
                            
                            # 6. 调用LLaVA生成描述
                            llava_start = time.time()
                            description = describe_image_with_position(
                                target_image,  # 目标特写
                                annotated_image,  # 带标注的完整图像
                                position_desc,  # 位置信息
                                simple_output=True
                            )
                            llava_time = time.time() - llava_start
                            
                            # 7. 处理描述结果
                            language = description.strip()
                            
                            # 保存语言描述到文件（在原始路径保存为language_llava.txt）
                            os.makedirs(os.path.dirname(language_file_llava), exist_ok=True)
                            with open(language_file_llava, 'w', encoding='utf-8') as f:
                                f.write(language)
                                
                            print(f"  - LLaVA描述已保存到: {language_file_llava}")
                        else:
                            print(f"  - 警告: 边界框无效，使用原始描述")
                            language = self._get_original_language(original_language_file)
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
        
        # 计算总耗时
        total_time = time.time() - start_time
        #print(f"  - 序列处理总耗时: {total_time:.2f}秒")
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
        return len(self.sequence_list)

    def __del__(self):
        """清理资源"""
        if LLAVA_AVAILABLE:
            try:
                cleanup_llava_model()
            except:
                pass

    @property
    def sequence_list(self):
        """获取序列列表"""
        return self._get_sequence_list()

    def _get_sequence_list(self):
        sequence_list = sorted([p.split('/')[-2] for p in glob.glob(os.path.join(self.base_path, '*/'))])
        return sequence_list
