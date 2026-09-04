import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
from PIL import Image, ImageDraw
import os
import sys
import time
import jieba
from collections import Counter
import re  # 添加正则表达式模块

# 将llava.py所在目录添加到Python路径
llava_path = os.path.dirname(os.path.abspath(__file__))
llava_path = os.path.join(llava_path, "..", "..")
sys.path.insert(0, llava_path)

# 导入llava函数
try:
    from llava import describe_image_with_position, describe_image, describe_bbox_target, cleanup_llava_model
    LLAVA_AVAILABLE = True
    print("OTBDataset: LLaVA模块可用")
except ImportError as e:
    LLAVA_AVAILABLE = False
    print(f"OTBDataset: LLaVA模块不可用: {e}")


class OTBDataset(BaseDataset):
    """ OTB-2015 dataset - 三种描述模态融合版 """
    def __init__(self, use_fusion_description=True):
        """
        初始化OTB数据集
        Args:
            use_fusion_description: 是否使用融合描述（True=使用，False=使用基础描述）
        """
        super().__init__()
        self.base_path = self.env_settings.otb_path
        self.sequence_info_list = self._get_sequence_info_list()
        self.use_fusion_description = use_fusion_description
        
        # 确保语言描述目录存在
        self.language_dir = os.path.join(self.base_path, 'llava_fusion')
        if not os.path.exists(self.language_dir):
            os.makedirs(self.language_dir, exist_ok=True)
            
        # 初始化jieba分词器
        jieba.initialize()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

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

    def _fuse_descriptions(self, descriptions):
        """
        融合三种描述
        Args:
            descriptions: 三种描述的列表 [模板+图像描述, 纯模板描述, 标注描述]
        Returns:
            融合后的描述文本
        """
        if len(descriptions) < 3:
            if descriptions:
                # 如果描述不足3个，返回第一个描述并去除多余空格
                result = descriptions[0].strip()
                return re.sub(r'\s+', ' ', result)
            else:
                return ""
        
        # 1. 处理每个描述的原始文本，去除多余空格
        cleaned_descriptions = []
        for desc in descriptions:
            if desc:
                # 去除两端空格，将多个连续空格替换为单个空格
                cleaned_desc = re.sub(r'\s+', ' ', desc.strip())
                cleaned_descriptions.append(cleaned_desc)
            else:
                cleaned_descriptions.append("")
        
        # 判断是否为英文描述
        def is_english(text):
            # 简单的英文检测：如果包含英文字母且中文比例较低，则认为是英文
            english_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z' or c == ' ')
            total_chars = len(text.replace(' ', ''))
            if total_chars == 0:
                return False
            return english_chars / total_chars > 0.5
        
        # 检查第一个描述是否为英文
        if is_english(cleaned_descriptions[0]):
            # 英文描述处理：使用空格分词而不是jieba
            all_words = []
            for sentence in cleaned_descriptions:
                words = sentence.split()
                all_words.extend(words)
            
            # 统计每个词的出现频率
            word_counts = Counter(all_words)
            
            # 以第一个描述为基础顺序，筛选出出现次数>=2的词
            base_words = cleaned_descriptions[0].split()
            final_words = []
            
            for word in base_words:
                # 过滤掉空词
                if word and word_counts.get(word, 0) >= 2:
                    final_words.append(word)
            
            # 组合成最终描述
            final_description = " ".join(final_words)
        else:
            # 中文描述处理：使用jieba分词
            # 2. 对所有描述进行分词
            all_words = []
            for sentence in cleaned_descriptions:
                words = jieba.lcut(sentence)
                all_words.extend(words)
            
            # 3. 统计每个词的出现频率
            word_counts = Counter(all_words)
            
            # 4. 以第一个描述（模板+图像）为基础顺序，筛选出出现次数>=2的词
            base_words = jieba.lcut(cleaned_descriptions[0])
            final_words = []
            
            for word in base_words:
                # 过滤掉纯空格的词
                if word.strip() and word_counts.get(word, 0) >= 2:
                    final_words.append(word)
            
            # 5. 组合成最终描述
            # 对于中文，使用空字符串连接，因为中文通常不需要词间空格
            final_description = "".join(final_words)
        
        # 如果融合后为空，使用第一个描述
        if not final_description.strip():
            final_description = cleaned_descriptions[0]
        
        # 最后清理多余的空格
        final_description = re.sub(r'\s+', ' ', final_description.strip())
        
        # 处理英文标点符号前的空格
        final_description = re.sub(r'\s+([,.!?;:])', r'\1', final_description)
        
        return final_description

    def _construct_sequence(self, sequence_info):
        """构建序列 - 融合三种描述方式"""
        # 记录开始时间
        start_time = time.time()
        
        # 基本序列信息
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        # 初始化省略帧
        init_omit = sequence_info.get('initOmit', 0)

        # 构建第一帧图片路径
        first_frame_num = start_frame + init_omit
        first_frame_path = '{base_path}/OTB_videos/{sequence_path}/{frame:0{nz}}.{ext}'.format(
            base_path=self.base_path, 
            sequence_path=sequence_path, 
            frame=first_frame_num, 
            nz=nz, 
            ext=ext
        )
        
        # 构建所有帧的路径列表
        frames = ['{base_path}/OTB_videos/{sequence_path}/{frame:0{nz}}.{ext}'.format(
            base_path=self.base_path, 
            sequence_path=sequence_path, 
            frame=frame_num, 
            nz=nz, 
            ext=ext
        ) for frame_num in range(start_frame + init_omit, end_frame + 1)]

        # 加载标注文件
        anno_path = '{}/{}/{}'.format(self.base_path, 'OTB_videos', sequence_info['anno_path'])
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')
        
        # 获取第一帧的边界框
        first_bbox = ground_truth_rect[init_omit] if len(ground_truth_rect) > init_omit else ground_truth_rect[0]
        
        # 语言描述生成
        language = ""
        
        # 检查是否有缓存的融合描述
        language_file_path = os.path.join(self.language_dir, f"{sequence_info['name']}.txt")
        if os.path.exists(language_file_path):
            with open(language_file_path, 'r', encoding='utf-8') as f:
                language = f.readlines()[0].rstrip()
                print(f"[OTBDataset] 加载缓存的融合描述: {sequence_info['name']}")
        
        # 如果没有缓存且LLaVA可用，生成三种描述并融合
        elif self.use_fusion_description and LLAVA_AVAILABLE and os.path.exists(first_frame_path):
            try:
                print(f"\n[OTBDataset] 处理序列: {sequence_info['name']}")
                
                # 1. 加载原始图像
                image = Image.open(first_frame_path).convert('RGB')
                img_width, img_height = image.size
                
                # 2. 获取三种描述
                descriptions = []
                
                # 描述1: 模板+图像描述（基础描述）
                try:
                    # 裁剪目标区域
                    x, y, w, h = first_bbox
                    x1 = max(0, int(x))
                    y1 = max(0, int(y))
                    x2 = min(img_width, int(x + w))
                    y2 = min(img_height, int(y + h))
                    
                    if x2 > x1 and y2 > y1 and w > 0 and h > 0:
                        # 裁剪目标
                        target_image = image.crop((x1, y1, x2, y2))
                        
                        # 在原始图像上绘制边界框
                        annotated_image = image.copy()
                        draw = ImageDraw.Draw(annotated_image)
                        draw.rectangle([x, y, x + w, y + h], outline='red', width=3)
                        
                        # 生成位置描述
                        position_desc = self._get_position_description(first_bbox, img_width, img_height)
                        
                        # 调用LLaVA生成模板+图像描述
                        description1 = describe_image_with_position(
                            target_image,  # 目标特写
                            annotated_image,  # 带标注的完整图像
                            position_desc,  # 位置信息
                            simple_output=True
                        )
                        descriptions.append(description1.strip())
                        print(f"  - 模板+图像描述生成成功")
                except Exception as e:
                    print(f"  - 模板+图像描述失败: {e}")
                
                # 描述2: 纯模板描述
                try:
                    if x2 > x1 and y2 > y1:
                        # 保存目标图像到临时文件
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                            target_image.save(tmp.name, 'JPEG', quality=95)
                            temp_target_path = tmp.name
                        
                        # 调用LLaVA分析目标图像
                        description2 = describe_image(temp_target_path, simple_output=True)
                        
                        # 提取描述内容
                        if description2.startswith("DESCRIPTION:"):
                            description2 = description2[len("DESCRIPTION:"):].strip()
                        
                        descriptions.append(description2.strip())
                        
                        # 清理临时文件
                        try:
                            os.unlink(temp_target_path)
                        except:
                            pass
                        print(f"  - 纯模板描述生成成功")
                except Exception as e:
                    print(f"  - 纯模板描述失败: {e}")
                
                # 描述3: 标注描述
                try:
                    # 创建标注图像
                    annotated_image = image.copy()
                    draw = ImageDraw.Draw(annotated_image)
                    draw.rectangle([x, y, x + w, y + h], outline='red', width=3)
                    
                    # 保存标注图像
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        annotated_image.save(tmp.name, 'JPEG')
                        temp_annotated_path = tmp.name
                    
                    # 使用带标注的图像调用LLaVA
                    description3 = describe_bbox_target(temp_annotated_path, first_bbox, simple_output=True)
                    
                    if description3.startswith("DESCRIPTION:"):
                        description3 = description3[len("DESCRIPTION:"):].strip()
                    
                    descriptions.append(description3.strip())
                    
                    # 清理临时文件
                    try:
                        os.unlink(temp_annotated_path)
                    except:
                        pass
                    print(f"  - 标注描述生成成功")
                except Exception as e:
                    print(f"  - 标注描述失败: {e}")
                
                # 3. 融合三种描述
                if len(descriptions) >= 2:
                    print(f"  - 原始描述: {descriptions}")
                    language = self._fuse_descriptions(descriptions)
                    print(f"  - 融合后描述: {language}")
                    
                    # 保存融合描述到文件
                    with open(language_file_path, 'w', encoding='utf-8') as f:
                        f.write(language)
                else:
                    # 如果只有一种描述可用，使用它
                    language = descriptions[0] if descriptions else self._get_fallback_language(sequence_info)
                    
            except Exception as e:
                print(f"  - 融合处理失败: {e}")
                language = self._get_fallback_language(sequence_info)
        else:
            # 使用后备描述
            language = self._get_fallback_language(sequence_info)
        
        # 计算总耗时
        total_time = time.time() - start_time
        print("========================")
        print(f"序列: {sequence_info['name']}")
        print(f"描述: {language}")
        print(f"耗时: {total_time:.2f}秒")
        print("========================")
        
        # 返回序列
        return Sequence(
            sequence_info['name'], 
            frames, 
            'otb', 
            ground_truth_rect[init_omit:,:],  # 原始边界框数据
            object_class=sequence_info['object_class'], 
            language=language  # 融合后的语言描述
        )

    def _get_fallback_language(self, sequence_info):
        """获取后备的语言描述"""
        # 尝试从模板+标注目录获取
        language_file = os.path.join(self.base_path, 'llava_模板+标注', f"{sequence_info['name']}.txt")
        if os.path.exists(language_file):
            with open(language_file, 'r', encoding='utf-8') as f:
                language = f.readlines()[0].rstrip()
                return language
        
        # 尝试从OTB_query_test目录获取
        language_file = os.path.join(self.base_path, 'OTB_query_test', f"{sequence_info['name']}.txt")
        if os.path.exists(language_file):
            with open(language_file, 'r', encoding='utf-8') as f:
                language = f.readlines()[0].rstrip()
                return language
        else:
            # 如果文件不存在，使用对象类别
            return sequence_info['object_class']

    def __len__(self):
        return len(self.sequence_info_list)

    def __del__(self):
        """清理资源"""
        if LLAVA_AVAILABLE:
            try:
                cleanup_llava_model()
            except:
                pass


    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "Basketball", "path": "Basketball/img", "startFrame": 1, "endFrame": 725, "nz": 4, "ext": "jpg", "anno_path": "Basketball/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Biker", "path": "Biker/img", "startFrame": 1, "endFrame": 142, "nz": 4, "ext": "jpg", "anno_path": "Biker/groundtruth_rect.txt",
             "object_class": "person head"},
            {"name": "Bird1", "path": "Bird1/img", "startFrame": 1, "endFrame": 408, "nz": 4, "ext": "jpg", "anno_path": "Bird1/groundtruth_rect.txt",
             "object_class": "bird"},
            {"name": "Bird2", "path": "Bird2/img", "startFrame": 1, "endFrame": 99, "nz": 4, "ext": "jpg", "anno_path": "Bird2/groundtruth_rect.txt",
             "object_class": "bird"},
            {"name": "BlurBody", "path": "BlurBody/img", "startFrame": 1, "endFrame": 334, "nz": 4, "ext": "jpg", "anno_path": "BlurBody/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "BlurCar1", "path": "BlurCar1/img", "startFrame": 247, "endFrame": 988, "nz": 4, "ext": "jpg", "anno_path": "BlurCar1/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "BlurCar2", "path": "BlurCar2/img", "startFrame": 1, "endFrame": 585, "nz": 4, "ext": "jpg", "anno_path": "BlurCar2/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "BlurCar3", "path": "BlurCar3/img", "startFrame": 3, "endFrame": 359, "nz": 4, "ext": "jpg", "anno_path": "BlurCar3/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "BlurCar4", "path": "BlurCar4/img", "startFrame": 18, "endFrame": 397, "nz": 4, "ext": "jpg", "anno_path": "BlurCar4/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "BlurFace", "path": "BlurFace/img", "startFrame": 1, "endFrame": 493, "nz": 4, "ext": "jpg", "anno_path": "BlurFace/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "BlurOwl", "path": "BlurOwl/img", "startFrame": 1, "endFrame": 631, "nz": 4, "ext": "jpg", "anno_path": "BlurOwl/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Board", "path": "Board/img", "startFrame": 1, "endFrame": 698, "nz": 4, "ext": "jpg", "anno_path": "Board/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Bolt", "path": "Bolt/img", "startFrame": 1, "endFrame": 350, "nz": 4, "ext": "jpg", "anno_path": "Bolt/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Bolt2", "path": "Bolt2/img", "startFrame": 1, "endFrame": 293, "nz": 4, "ext": "jpg", "anno_path": "Bolt2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Box", "path": "Box/img", "startFrame": 1, "endFrame": 1161, "nz": 4, "ext": "jpg", "anno_path": "Box/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Boy", "path": "Boy/img", "startFrame": 1, "endFrame": 602, "nz": 4, "ext": "jpg", "anno_path": "Boy/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Car1", "path": "Car1/img", "startFrame": 1, "endFrame": 1020, "nz": 4, "ext": "jpg", "anno_path": "Car1/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "Car2", "path": "Car2/img", "startFrame": 1, "endFrame": 913, "nz": 4, "ext": "jpg", "anno_path": "Car2/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "Car24", "path": "Car24/img", "startFrame": 1, "endFrame": 3059, "nz": 4, "ext": "jpg", "anno_path": "Car24/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "Car4", "path": "Car4/img", "startFrame": 1, "endFrame": 659, "nz": 4, "ext": "jpg", "anno_path": "Car4/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "CarDark", "path": "CarDark/img", "startFrame": 1, "endFrame": 393, "nz": 4, "ext": "jpg", "anno_path": "CarDark/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "CarScale", "path": "CarScale/img", "startFrame": 1, "endFrame": 252, "nz": 4, "ext": "jpg", "anno_path": "CarScale/groundtruth_rect.txt",
             "object_class": "car"},
            #{"name": "ClifBar", "path": "ClifBar/img", "startFrame": 1, "endFrame": 472, "nz": 4, "ext": "jpg", "anno_path": "ClifBar/groundtruth_rect.txt",
            # "object_class": "other"},
            {"name": "Coke", "path": "Coke/img", "startFrame": 1, "endFrame": 291, "nz": 4, "ext": "jpg", "anno_path": "Coke/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Couple", "path": "Couple/img", "startFrame": 1, "endFrame": 140, "nz": 4, "ext": "jpg", "anno_path": "Couple/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Coupon", "path": "Coupon/img", "startFrame": 1, "endFrame": 327, "nz": 4, "ext": "jpg", "anno_path": "Coupon/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Crossing", "path": "Crossing/img", "startFrame": 1, "endFrame": 120, "nz": 4, "ext": "jpg", "anno_path": "Crossing/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Crowds", "path": "Crowds/img", "startFrame": 1, "endFrame": 347, "nz": 4, "ext": "jpg", "anno_path": "Crowds/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Dancer", "path": "Dancer/img", "startFrame": 1, "endFrame": 225, "nz": 4, "ext": "jpg", "anno_path": "Dancer/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Dancer2", "path": "Dancer2/img", "startFrame": 1, "endFrame": 150, "nz": 4, "ext": "jpg", "anno_path": "Dancer2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "David", "path": "David/img", "startFrame": 300, "endFrame": 770, "nz": 4, "ext": "jpg", "anno_path": "David/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "David2", "path": "David2/img", "startFrame": 1, "endFrame": 537, "nz": 4, "ext": "jpg", "anno_path": "David2/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "David3", "path": "David3/img", "startFrame": 1, "endFrame": 252, "nz": 4, "ext": "jpg", "anno_path": "David3/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Deer", "path": "Deer/img", "startFrame": 1, "endFrame": 71, "nz": 4, "ext": "jpg", "anno_path": "Deer/groundtruth_rect.txt",
             "object_class": "mammal"},
            {"name": "Diving", "path": "Diving/img", "startFrame": 1, "endFrame": 215, "nz": 4, "ext": "jpg", "anno_path": "Diving/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Dog", "path": "Dog/img", "startFrame": 1, "endFrame": 127, "nz": 4, "ext": "jpg", "anno_path": "Dog/groundtruth_rect.txt",
             "object_class": "dog"},
            {"name": "Dog1", "path": "Dog1/img", "startFrame": 1, "endFrame": 1350, "nz": 4, "ext": "jpg", "anno_path": "Dog1/groundtruth_rect.txt",
             "object_class": "dog"},
            {"name": "Doll", "path": "Doll/img", "startFrame": 1, "endFrame": 3872, "nz": 4, "ext": "jpg", "anno_path": "Doll/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "DragonBaby", "path": "DragonBaby/img", "startFrame": 1, "endFrame": 113, "nz": 4, "ext": "jpg", "anno_path": "DragonBaby/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Dudek", "path": "Dudek/img", "startFrame": 1, "endFrame": 1145, "nz": 4, "ext": "jpg", "anno_path": "Dudek/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "FaceOcc1", "path": "FaceOcc1/img", "startFrame": 1, "endFrame": 892, "nz": 4, "ext": "jpg", "anno_path": "FaceOcc1/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "FaceOcc2", "path": "FaceOcc2/img", "startFrame": 1, "endFrame": 812, "nz": 4, "ext": "jpg", "anno_path": "FaceOcc2/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Fish", "path": "Fish/img", "startFrame": 1, "endFrame": 476, "nz": 4, "ext": "jpg", "anno_path": "Fish/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "FleetFace", "path": "FleetFace/img", "startFrame": 1, "endFrame": 707, "nz": 4, "ext": "jpg", "anno_path": "FleetFace/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Football", "path": "Football/img", "startFrame": 1, "endFrame": 362, "nz": 4, "ext": "jpg", "anno_path": "Football/groundtruth_rect.txt",
             "object_class": "person head"},
            {"name": "Football1", "path": "Football1/img", "startFrame": 1, "endFrame": 74, "nz": 4, "ext": "jpg", "anno_path": "Football1/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Freeman1", "path": "Freeman1/img", "startFrame": 1, "endFrame": 326, "nz": 4, "ext": "jpg", "anno_path": "Freeman1/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Freeman3", "path": "Freeman3/img", "startFrame": 1, "endFrame": 460, "nz": 4, "ext": "jpg", "anno_path": "Freeman3/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Freeman4", "path": "Freeman4/img", "startFrame": 1, "endFrame": 283, "nz": 4, "ext": "jpg", "anno_path": "Freeman4/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Girl", "path": "Girl/img", "startFrame": 1, "endFrame": 500, "nz": 4, "ext": "jpg", "anno_path": "Girl/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Girl2", "path": "Girl2/img", "startFrame": 1, "endFrame": 1500, "nz": 4, "ext": "jpg", "anno_path": "Girl2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Gym", "path": "Gym/img", "startFrame": 1, "endFrame": 767, "nz": 4, "ext": "jpg", "anno_path": "Gym/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human2", "path": "Human2/img", "startFrame": 1, "endFrame": 1128, "nz": 4, "ext": "jpg", "anno_path": "Human2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human3", "path": "Human3/img", "startFrame": 1, "endFrame": 1698, "nz": 4, "ext": "jpg", "anno_path": "Human3/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human4", "path": "Human4/img", "startFrame": 1, "endFrame": 667, "nz": 4, "ext": "jpg", "anno_path": "Human4/groundtruth_rect.2.txt",
             "object_class": "person"},
            {"name": "Human5", "path": "Human5/img", "startFrame": 1, "endFrame": 713, "nz": 4, "ext": "jpg", "anno_path": "Human5/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human6", "path": "Human6/img", "startFrame": 1, "endFrame": 792, "nz": 4, "ext": "jpg", "anno_path": "Human6/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human7", "path": "Human7/img", "startFrame": 1, "endFrame": 250, "nz": 4, "ext": "jpg", "anno_path": "Human7/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human8", "path": "Human8/img", "startFrame": 1, "endFrame": 128, "nz": 4, "ext": "jpg", "anno_path": "Human8/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human9", "path": "Human9/img", "startFrame": 1, "endFrame": 305, "nz": 4, "ext": "jpg", "anno_path": "Human9/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Ironman", "path": "Ironman/img", "startFrame": 1, "endFrame": 166, "nz": 4, "ext": "jpg", "anno_path": "Ironman/groundtruth_rect.txt",
             "object_class": "person head"},
            {"name": "Jogging-1", "path": "Jogging/img", "startFrame": 1, "endFrame": 307, "nz": 4, "ext": "jpg", "anno_path": "Jogging/groundtruth_rect.1.txt",
             "object_class": "person"},
            {"name": "Jogging-2", "path": "Jogging/img", "startFrame": 1, "endFrame": 307, "nz": 4, "ext": "jpg", "anno_path": "Jogging/groundtruth_rect.2.txt",
             "object_class": "person"},
            {"name": "Jump", "path": "Jump/img", "startFrame": 1, "endFrame": 122, "nz": 4, "ext": "jpg", "anno_path": "Jump/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Jumping", "path": "Jumping/img", "startFrame": 1, "endFrame": 313, "nz": 4, "ext": "jpg", "anno_path": "Jumping/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "KiteSurf", "path": "KiteSurf/img", "startFrame": 1, "endFrame": 84, "nz": 4, "ext": "jpg", "anno_path": "KiteSurf/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Lemming", "path": "Lemming/img", "startFrame": 1, "endFrame": 1336, "nz": 4, "ext": "jpg", "anno_path": "Lemming/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Liquor", "path": "Liquor/img", "startFrame": 1, "endFrame": 1741, "nz": 4, "ext": "jpg", "anno_path": "Liquor/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Man", "path": "Man/img", "startFrame": 1, "endFrame": 134, "nz": 4, "ext": "jpg", "anno_path": "Man/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Matrix", "path": "Matrix/img", "startFrame": 1, "endFrame": 100, "nz": 4, "ext": "jpg", "anno_path": "Matrix/groundtruth_rect.txt",
             "object_class": "person head"},
            {"name": "Mhyang", "path": "Mhyang/img", "startFrame": 1, "endFrame": 1490, "nz": 4, "ext": "jpg", "anno_path": "Mhyang/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "MotorRolling", "path": "MotorRolling/img", "startFrame": 1, "endFrame": 164, "nz": 4, "ext": "jpg", "anno_path": "MotorRolling/groundtruth_rect.txt",
             "object_class": "vehicle"},
            {"name": "MountainBike", "path": "MountainBike/img", "startFrame": 1, "endFrame": 228, "nz": 4, "ext": "jpg", "anno_path": "MountainBike/groundtruth_rect.txt",
             "object_class": "bicycle"},
            {"name": "Panda", "path": "Panda/img", "startFrame": 1, "endFrame": 1000, "nz": 4, "ext": "jpg", "anno_path": "Panda/groundtruth_rect.txt",
             "object_class": "mammal"},
            {"name": "RedTeam", "path": "RedTeam/img", "startFrame": 1, "endFrame": 1918, "nz": 4, "ext": "jpg", "anno_path": "RedTeam/groundtruth_rect.txt",
             "object_class": "vehicle"},
            {"name": "Rubik", "path": "Rubik/img", "startFrame": 1, "endFrame": 1997, "nz": 4, "ext": "jpg", "anno_path": "Rubik/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Shaking", "path": "Shaking/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg", "anno_path": "Shaking/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Singer1", "path": "Singer1/img", "startFrame": 1, "endFrame": 351, "nz": 4, "ext": "jpg", "anno_path": "Singer1/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Singer2", "path": "Singer2/img", "startFrame": 1, "endFrame": 366, "nz": 4, "ext": "jpg", "anno_path": "Singer2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Skater", "path": "Skater/img", "startFrame": 1, "endFrame": 160, "nz": 4, "ext": "jpg", "anno_path": "Skater/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Skater2", "path": "Skater2/img", "startFrame": 1, "endFrame": 435, "nz": 4, "ext": "jpg", "anno_path": "Skater2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Skating1", "path": "Skating1/img", "startFrame": 1, "endFrame": 400, "nz": 4, "ext": "jpg", "anno_path": "Skating1/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Skating2-1", "path": "Skating2-1/img", "startFrame": 1, "endFrame": 473, "nz": 4, "ext": "jpg", "anno_path": "Skating2-1/groundtruth_rect.1.txt",
             "object_class": "person"},
            {"name": "Skating2-2", "path": "Skating2-2/img", "startFrame": 1, "endFrame": 473, "nz": 4, "ext": "jpg", "anno_path": "Skating2-2/groundtruth_rect.2.txt",
             "object_class": "person"},
            {"name": "Skiing", "path": "Skiing/img", "startFrame": 1, "endFrame": 81, "nz": 4, "ext": "jpg", "anno_path": "Skiing/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Soccer", "path": "Soccer/img", "startFrame": 1, "endFrame": 392, "nz": 4, "ext": "jpg", "anno_path": "Soccer/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Subway", "path": "Subway/img", "startFrame": 1, "endFrame": 175, "nz": 4, "ext": "jpg", "anno_path": "Subway/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Surfer", "path": "Surfer/img", "startFrame": 1, "endFrame": 376, "nz": 4, "ext": "jpg", "anno_path": "Surfer/groundtruth_rect.txt",
             "object_class": "person head"},
            {"name": "Suv", "path": "Suv/img", "startFrame": 1, "endFrame": 945, "nz": 4, "ext": "jpg", "anno_path": "Suv/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "Sylvester", "path": "Sylvester/img", "startFrame": 1, "endFrame": 1345, "nz": 4, "ext": "jpg", "anno_path": "Sylvester/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Tiger1", "path": "Tiger1/img", "startFrame": 1, "endFrame": 354, "nz": 4, "ext": "jpg", "anno_path": "Tiger1/groundtruth_rect.txt", "initOmit": 5,
             "object_class": "other"},
            {"name": "Tiger2", "path": "Tiger2/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg", "anno_path": "Tiger2/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Toy", "path": "Toy/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg", "anno_path": "Toy/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Trans", "path": "Trans/img", "startFrame": 1, "endFrame": 124, "nz": 4, "ext": "jpg", "anno_path": "Trans/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Trellis", "path": "Trellis/img", "startFrame": 1, "endFrame": 569, "nz": 4, "ext": "jpg", "anno_path": "Trellis/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Twinnings", "path": "Twinnings/img", "startFrame": 1, "endFrame": 472, "nz": 4, "ext": "jpg", "anno_path": "Twinnings/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Vase", "path": "Vase/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg", "anno_path": "Vase/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Walking", "path": "Walking/img", "startFrame": 1, "endFrame": 412, "nz": 4, "ext": "jpg", "anno_path": "Walking/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Walking2", "path": "Walking2/img", "startFrame": 1, "endFrame": 500, "nz": 4, "ext": "jpg", "anno_path": "Walking2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Woman", "path": "Woman/img", "startFrame": 1, "endFrame": 597, "nz": 4, "ext": "jpg", "anno_path": "Woman/groundtruth_rect.txt",
             "object_class": "person"}
        ]
    
        return sequence_info_list

