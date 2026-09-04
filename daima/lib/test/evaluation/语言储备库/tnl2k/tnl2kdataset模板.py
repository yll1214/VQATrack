import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import glob
import os
import sys
from PIL import Image

# 将llava.py所在目录添加到Python路径
llava_path = os.path.dirname(os.path.abspath(__file__))
llava_path = os.path.join(llava_path, "..", "..")  # 向上两级到项目根目录
sys.path.insert(0, llava_path)

# 导入llava的describe_image函数
try:
    from llava import describe_image
    LLAVA_AVAILABLE = True
    print("LLaVA模块加载成功")
except ImportError as e:
    LLAVA_AVAILABLE = False
    print(f"LLaVA模块加载失败: {e}")


class TNL2KDataset(BaseDataset):
    """
    TNL2K dataset
    """
    def __init__(self, generate_language=True):
        """
        Args:
            generate_language: 是否一次性生成所有语言描述
        """
        super().__init__()
        self.base_path = self.env_settings.tnl2k_path
        
        # 如果指定了生成语言，则一次性生成所有序列的语言描述
        if generate_language and LLAVA_AVAILABLE:
            print("开始一次性生成所有序列的语言描述...")
            self._generate_all_language_descriptions()
            print("所有序列的语言描述生成完成！")

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        """构造序列，直接使用已生成的语言文件"""
        anno_path = os.path.join(self.base_path, sequence_name, 'groundtruth.txt')
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_list = sorted(glob.glob(os.path.join(self.base_path, sequence_name, 'imgs', '*')))

        # 原始语言文件路径
        original_language_file = os.path.join(self.base_path, sequence_name, "language.txt")
        # LLaVA生成的语言文件路径（保存在原始路径）
        language_file_llava = os.path.join(self.base_path, sequence_name, "language_模板.txt")
        
        # 读取语言描述
        language = ""
        if os.path.exists(language_file_llava):
            with open(language_file_llava, 'r', encoding='utf-8') as f:
                language = f.readlines()[0].rstrip()
        elif os.path.exists(original_language_file):
            with open(original_language_file, 'r', encoding='utf-8') as f:
                language = f.readlines()[0].rstrip()
        
        return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4),
                        object_class=None, target_visible=None, language=language)

    def __len__(self):
        return len(self.sequence_list)

    def _generate_all_language_descriptions(self):
        """一次性生成所有序列的语言描述"""
        sequences = self._get_sequence_list()
        
        for i, sequence_name in enumerate(sequences):
            # 检查是否已经生成过
            language_file_llava = os.path.join(self.base_path, sequence_name, "language_llava.txt")
            if os.path.exists(language_file_llava):
                print(f"[{i+1}/{len(sequences)}] 跳过已生成的: {sequence_name}")
                continue
                
            try:
                print(f"\n[{i+1}/{len(sequences)}] 处理序列: {sequence_name}")
                
                # 构造序列基本信息
                anno_path = os.path.join(self.base_path, sequence_name, 'groundtruth.txt')
                frames_dir = os.path.join(self.base_path, sequence_name, 'imgs')
                
                if not os.path.exists(anno_path) or not os.path.exists(frames_dir):
                    print(f"  警告: {sequence_name} 文件不存在，跳过")
                    continue
                
                # 读取第一帧的ground truth
                ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
                if len(ground_truth_rect) == 0:
                    print(f"  警告: {sequence_name} 没有ground truth数据，跳过")
                    continue
                
                # 获取第一帧图片
                frames_list = sorted(glob.glob(os.path.join(frames_dir, '*')))
                if len(frames_list) == 0:
                    print(f"  警告: {sequence_name} 没有图片，跳过")
                    continue
                
                first_frame_path = frames_list[0]
                if not os.path.exists(first_frame_path):
                    print(f"  警告: {first_frame_path} 不存在，跳过")
                    continue
                
                # 获取第一帧的边界框
                first_bbox = ground_truth_rect[0]
                
                # 裁剪目标区域图像
                image = Image.open(first_frame_path).convert('RGB')
                img_width, img_height = image.size
                
                # 裁剪目标区域
                x, y, w, h = first_bbox
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(img_width, int(x + w))
                y2 = min(img_height, int(y + h))
                
                if x2 > x1 and y2 > y1:
                    target_image = image.crop((x1, y1, x2, y2))
                    
                    # 保存目标图像到临时文件
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        target_image.save(tmp.name, 'JPEG', quality=95)
                        temp_target_path = tmp.name
                    
                    # 使用LLaVA分析目标图像
                    print(f"  使用LLaVA分析目标...")
                    description = describe_image(temp_target_path, simple_output=True)
                    
                    # 提取描述内容
                    if description.startswith("DESCRIPTION:"):
                        description = description[len("DESCRIPTION:"):].strip()
                    
                    # 保存语言描述到文件（language_llava.txt）
                    os.makedirs(os.path.dirname(language_file_llava), exist_ok=True)
                    with open(language_file_llava, 'w', encoding='utf-8') as f:
                        f.write(description)
                    
                    print(f"  LLaVA描述已保存到: {language_file_llava}")
                    
                    # 清理临时文件
                    try:
                        os.unlink(temp_target_path)
                    except:
                        pass
                else:
                    print(f"  边界框无效，跳过")
                    
            except Exception as e:
                print(f"  处理 {sequence_name} 时出错: {e}")
                # 如果失败，使用原始描述
                original_language_file = os.path.join(self.base_path, sequence_name, "language.txt")
                if os.path.exists(original_language_file):
                    with open(original_language_file, 'r', encoding='utf-8') as f:
                        language = f.readlines()[0].rstrip()
                    with open(language_file_llava, 'w', encoding='utf-8') as f:
                        f.write(language)
                    print(f"  使用原始描述: {language}")
                continue

    @property
    def sequence_list(self):
        """获取序列列表"""
        return self._get_sequence_list()

    def _get_sequence_list(self):
        sequence_list = sorted([p.split('/')[-2] for p in glob.glob(os.path.join(self.base_path, '*/'))])
        return sequence_list
