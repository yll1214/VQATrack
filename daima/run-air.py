import importlib
import cv2
import argparse
import time
import numpy as np
import yaml
import os
import airsim
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
import scipy.ndimage as ndimage
import math
import subprocess
import tempfile
import sys

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Demo with AirSim')
parser.add_argument('--tracker_name', type=str, default='LiteTrack')
parser.add_argument('--tracker_param', type=str, default='baseline_base')
parser.add_argument('--output_video', type=str, default='airsim_output.mp4')
parser.add_argument('--language', type=str, default='')
parser.add_argument('--camera_name', type=str, default='fpv', help='AirSim camera name')
parser.add_argument('--airsim_host', type=str, default='127.0.0.1', help='AirSim host IP')
parser.add_argument('--depth_model', type=str, default='vitb', choices=['vits', 'vitb', 'vitl'], 
                    help='Depth model encoder type')
parser.add_argument('--depth_checkpoint', type=str, default='checkpoints/depth_anything_v2_vitb.pth',
                    help='Path to depth model checkpoint')
parser.add_argument('--fov_degrees', type=float, default=90.0, help='Camera field of view in degrees')
parser.add_argument('--takeoff_height', type=float, default=2.0, help='Takeoff height in meters')
parser.add_argument('--control_enabled', action='store_true', default=True, 
                    help='Enable drone control to follow target')
parser.add_argument('--max_speed', type=float, default=0.5, 
                    help='Maximum drone speed in m/s')
parser.add_argument('--target_threshold', type=float, default=2.0, 
                    help='Distance threshold to target in meters')
parser.add_argument('--center_threshold', type=float, default=0.1, 
                    help='Center threshold as ratio of image size')
parser.add_argument('--llava_path', type=str, default='llava.py',
                    help='Path to llava.py script')
parser.add_argument('--llava_mode', type=str, default='target_with_bbox',
              choices=['target_only', 'target_with_bbox', 'target_with_scene'],
                    help='LLaVA analysis mode: target_only, target_with_bbox, or target_with_scene')
args = parser.parse_args()

# 加载 YAML 文件
try:
    tracker_param = os.path.join('experiments', args.tracker_name, args.tracker_param +'.yaml')
    with open(tracker_param, 'r') as file:
        tracker_param_dict = yaml.safe_load(file)
except yaml.YAMLError as e:
    print(f"解析参数文件时出错: {e}")
    exit()

# 提取测试模式
test_params = tracker_param_dict.get('TEST', {})
args.mode = test_params.get('MODE', 'NL')  # 默认为 BBOX 模式
print(f"跟踪模式设置为: {args.mode}")

class LLavaMode:
    TARGET_ONLY = "target_only"        # 只截取目标区域
    TARGET_WITH_BBOX = "target_with_bbox"  # 截取目标+标注框
    TARGET_WITH_SCENE = "target_with_scene"  # 截取目标+原始图像

class PIDController:
    def __init__(self, Kp, Ki, Kd, max_output=2.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.last_error = 0
        self.integral = 0
        self.max_output = max_output
        
    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        
        # 抗积分饱和
        if abs(self.integral) > 2.0:
            self.integral = 0
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        output = np.clip(output, -self.max_output, self.max_output)
        
        self.last_error = error
        return output

class ObstacleAvoidance:
    def __init__(self, drone_controller, client, image_width=640, image_height=360):
        self.drone_controller = drone_controller
        self.client = client
        self.image_width = image_width
        self.image_height = image_height
        
        # 控制参数
        self.pid_y = PIDController(Kp=1.0, Ki=0.01, Kd=0.2)
        
        # 飞行参数
        self.obstacle_threshold = 0.8
        self.safe_margin = 0.4
        self.avoid_speed_multiplier = 2.5
        
        # 状态变量
        self.target_y = 0
        self.last_avoid_time = 0
        self.avoid_cooldown = 0.001
        self.is_avoiding = False
        self.prev_pos = self.get_current_position()
        self.smoothing_factor = 0.2
        self.smoothed_y = self.prev_pos.y_val
        
        # 轨迹绘制
        self.trajectory_points = []

    def get_current_position(self):
        """获取当前无人机位置"""
        state = self.client.getMultirotorState()
        return state.kinematics_estimated.position
    
    def analyze_obstacles(self, depth_map):
        """分析障碍物 - 修复数据类型问题"""
        h, w = depth_map.shape
        
        # 确保深度图为float类型
        if depth_map.dtype != np.float32 and depth_map.dtype != np.float64:
            depth_map = depth_map.astype(np.float32)
        
        # 分析区域划分
        y_start, y_end = h//3, 2*h//3  # 只关注中间区域
        x_left = w//3
        x_right = 2*w//3
        
        # 计算各区域最小深度 - 使用numpy而不是torch
        left_region = depth_map[y_start:y_end, :x_left]
        center_region = depth_map[y_start:y_end, x_left:x_right]
        right_region = depth_map[y_start:y_end, x_right:]
        
        # 使用numpy计算分位数
        min_left = np.percentile(left_region, 10)  # 10%分位数
        min_center = np.percentile(center_region, 10)
        min_right = np.percentile(right_region, 10)
        
        return {
            'left': float(min_left),
            'center': float(min_center),
            'right': float(min_right)
        }

    def calculate_avoidance(self, obstacle_data, current_y):
        """计算避障策略"""
        safety_threshold = self.obstacle_threshold + self.safe_margin
        
        left_safe = obstacle_data['left'] > safety_threshold
        center_safe = obstacle_data['center'] > safety_threshold
        right_safe = obstacle_data['right'] > safety_threshold
        
        new_y = None
        
        if not center_safe:
            if left_safe and not right_safe:
                new_y = current_y - self.safe_margin * 1.2
            elif right_safe and not left_safe:
                new_y = current_y + self.safe_margin * 1.2
            elif left_safe and right_safe:
                if obstacle_data['left'] > obstacle_data['right']:
                    new_y = current_y - self.safe_margin
                else:
                    new_y = current_y + self.safe_margin
            else:
                if obstacle_data['left'] > obstacle_data['right']:
                    new_y = current_y - self.safe_margin * 1.5
                else:
                    new_y = current_y + self.safe_margin * 1.5
        elif not left_safe or not right_safe:
            if left_safe and not right_safe:
                new_y = current_y - self.safe_margin * 0.5
            elif right_safe and not left_safe:
                new_y = current_y + self.safe_margin * 0.5
        
        # 平滑处理
        if new_y is not None:
            self.smoothed_y = self.smoothed_y * (1 - self.smoothing_factor) + new_y * self.smoothing_factor
            return self.smoothed_y
            
        return None

    def execute_avoidance(self, depth_map, dt):
        """执行避障控制 - 修复深度图处理"""
        current_pos = self.get_current_position()
        
        # 将深度图转换为用于避障的格式 - 确保数据类型正确
        if depth_map is not None:
            # 确保深度图为float类型
            if depth_map.dtype != np.float32 and depth_map.dtype != np.float64:
                depth_for_avoidance = depth_map.astype(np.float32)
            else:
                depth_for_avoidance = depth_map.copy()
            
            # 归一化处理 (0-1范围)
            if depth_for_avoidance.max() > 1.0:
                depth_for_avoidance = depth_for_avoidance / depth_for_avoidance.max()
            
            # 转换为实际距离估计
            depth_for_avoidance = (1 - depth_for_avoidance) * 15
            
            # 分析障碍物
            obstacle_data = self.analyze_obstacles(depth_for_avoidance)
        else:
            # 如果没有深度图，假设前方安全
            obstacle_data = {'left': 10.0, 'center': 10.0, 'right': 10.0}
        
        # 避障决策
        if time.time() - self.last_avoid_time > self.avoid_cooldown:
            new_y = self.calculate_avoidance(obstacle_data, current_pos.y_val)
            if new_y is not None:
                self.target_y = new_y
                self.last_avoid_time = time.time()
                self.is_avoiding = True
                print(f"[避障] 检测到障碍物，调整Y坐标至: {self.target_y:.2f}")
                print(f"[避障] 深度数据 - 左:{obstacle_data['left']:.2f}, 中:{obstacle_data['center']:.2f}, 右:{obstacle_data['right']:.2f}")
            else:
                self.is_avoiding = False
        
        # 如果正在避障，执行避障控制
        if self.is_avoiding:
            y_error = self.target_y - current_pos.y_val
            y_control = self.pid_y.update(y_error, dt)
            
            # 限制最大速度变化
            max_velocity_change = 0.5
            current_velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity.y_val
            desired_velocity = y_control
            
            if abs(desired_velocity - current_velocity) > max_velocity_change:
                desired_velocity = current_velocity + np.sign(desired_velocity - current_velocity) * max_velocity_change
            
            # 执行避障控制 - 保持当前高度和前向速度
            self.client.moveByVelocityZAsync(
                self.drone_controller.max_speed * 0.9,  # 降低前向速度
                desired_velocity,
                -self.drone_controller.target_height,  # 保持高度
                dt,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(False, 0)
            )
            return True
        return False

    def update_trajectory(self):
        """更新并绘制轨迹"""
        current_pos = self.get_current_position()
        self.trajectory_points.append(current_pos)
        
        # 保持轨迹点数量合理
        if len(self.trajectory_points) > 100:
            self.trajectory_points.pop(0)
        
        # 绘制轨迹
        if len(self.trajectory_points) >= 2:
            self.client.simPlotLineStrip(
                self.trajectory_points,
                color_rgba=[1.0, 0.0, 0.0, 1.0],  # 红色轨迹
                thickness=3.0,
                is_persistent=True
            )

# 深度估计优化类
class DepthEstimator:
    def __init__(self, model, transform, device):
        self.model = model
        self.transform = transform
        self.device = device
        self.depth_cache = None
        self.cache_weight = 0.7
        
    def estimate_depth(self, frame):
        """深度估计方法，生成灰度深度图"""
        if self.model is None:
            return None
            
        try:
            # 转换为RGB格式
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            
            # 预处理
            downscale_factor = 1.0
            new_h, new_w = int(h * downscale_factor), int(w * downscale_factor)

            if downscale_factor < 1.0:
                img_downscaled = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                img_downscaled = img_rgb.copy()
                new_h, new_w = h, w

            # 填充到14的倍数
            pad_h = (14 - new_h % 14) % 14
            pad_w = (14 - new_w % 14) % 14
            padded_img = cv2.copyMakeBorder(img_downscaled, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

            # 推理
            tensor = self.transform(padded_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if 'cuda' in str(self.device):
                    with torch.amp.autocast('cuda', enabled=True):
                        depth = self.model(tensor)
                else:
                    depth = self.model(tensor)
                    
                depth = depth.squeeze().cpu().float().numpy()[:new_h, :new_w]

            if downscale_factor < 1.0:
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # 生成灰度深度图
            raw_depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_gray = ((1 - raw_depth_normalized) * 255).astype(np.uint8)
            
            # 平滑处理
            if self.depth_cache is not None:
                depth_gray = (self.cache_weight * self.depth_cache + 
                            (1 - self.cache_weight) * depth_gray).astype(np.uint8)
            
            self.depth_cache = depth_gray.copy()
            
            return depth_gray
            
        except Exception as e:
            print(f"深度估计失败: {str(e)}")
            return None

    def get_depth_visualization(self, depth_map):
        """获取深度图的可视化版本"""
        if depth_map is None:
            return None
        return depth_map
        
    def get_raw_depth_for_calculation(self, frame):
        """用于位置计算的原始深度值（0-1范围）"""
        if self.model is None:
            return None
            
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            
            downscale_factor = 1.0
            new_h, new_w = int(h * downscale_factor), int(w * downscale_factor)

            if downscale_factor < 1.0:
                img_downscaled = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                img_downscaled = img_rgb.copy()
                new_h, new_w = h, w

            pad_h = (14 - new_h % 14) % 14
            pad_w = (14 - new_w % 14) % 14
            padded_img = cv2.copyMakeBorder(img_downscaled, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

            tensor = self.transform(padded_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if 'cuda' in str(self.device):
                    with torch.amp.autocast('cuda', enabled=True):
                        depth = self.model(tensor)
                else:
                    depth = self.model(tensor)
                    
                depth = depth.squeeze().cpu().float().numpy()[:new_h, :new_w]

            if downscale_factor < 1.0:
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            
            return depth_normalized
            
        except Exception as e:
            print(f"原始深度估计失败: {str(e)}")
            return None

# 速度平滑器类
class VelocitySmoother:
    """简单的速度平滑器，用于减少卡顿"""
    def __init__(self, smooth_factor=0.4):
        self.smooth_factor = smooth_factor
        self.last_vx = 0
        self.last_vy = 0
        self.last_yaw_rate = 0
        
    def smooth_velocity(self, vx, vy, yaw_rate):
        """应用简单的指数平滑"""
        smooth_vx = self.smooth_factor * vx + (1 - self.smooth_factor) * self.last_vx
        smooth_vy = self.smooth_factor * vy + (1 - self.smooth_factor) * self.last_vy
        smooth_yaw = self.smooth_factor * yaw_rate + (1 - self.smooth_factor) * self.last_yaw_rate
        
        self.last_vx = smooth_vx
        self.last_vy = smooth_vy
        self.last_yaw_rate = smooth_yaw
        
        return smooth_vx, smooth_vy, smooth_yaw

# 位置解算优化类
class PositionSolver:
    def __init__(self, fov_degrees, image_width, image_height):
        self.fov_rad = np.radians(fov_degrees)
        self.image_width = image_width
        self.image_height = image_height
        
        # 计算焦距（像素单位）
        self.focal_length_x = image_width / (2 * np.tan(self.fov_rad / 2))
        self.focal_length_y = image_height / (2 * np.tan(self.fov_rad / 2))
        
        # 历史数据用于平滑
        self.position_history = []
        self.max_history = 10
        
    def calculate_world_coordinates(self, bbox, depth_map, camera_info):
        """世界坐标计算方法"""
        x, y, w, h = bbox
        
        # 计算边界框中心点和底部中心点
        center_x = x + w // 2
        center_y = y + h // 2
        bottom_center_y = y + h
        
        # 在边界框内采样多个点以获得更稳定的深度估计
        sample_points = [
            (center_x, center_y),  # 中心点
            (center_x, bottom_center_y),  # 底部中心点
            (x + w//4, center_y),  # 左侧点
            (x + 3*w//4, center_y),  # 右侧点
        ]
        
        # 计算采样点的平均深度
        valid_depths = []
        for px, py in sample_points:
            if (0 <= px < depth_map.shape[1] and 0 <= py < depth_map.shape[0] and
                depth_map[py, px] > 0.01):
                valid_depths.append(depth_map[py, px])
        
        if not valid_depths:
            return None, None, 0, 0
        
        # 使用中位数
        depth_value = np.median(valid_depths)
        
        # 深度值映射到实际距离
        min_depth = 0.3
        max_depth = 200.0
        actual_depth = min_depth + (max_depth - min_depth) * depth_value
        
        # 使用底部中心点计算位置
        pixel_x = center_x
        pixel_y = bottom_center_y
        
        # 将像素坐标转换为相机坐标系下的归一化坐标
        x_normalized = (pixel_x - self.image_width / 2) / self.focal_length_x
        y_normalized = -(pixel_y - self.image_height / 2) / self.focal_length_y
        
        # 相机坐标系中的3D点
        point_camera = np.array([
            actual_depth,
            x_normalized * actual_depth,
            y_normalized * actual_depth
        ])
        
        # 获取相机姿态（四元数）
        camera_pose = camera_info.pose
        qw = camera_pose.orientation.w_val
        qx = camera_pose.orientation.x_val
        qy = camera_pose.orientation.y_val
        qz = camera_pose.orientation.z_val
        
        # 将四元数转换为旋转矩阵
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        
        # 将相机坐标系中的点旋转到世界坐标系
        point_world_rotated = R @ point_camera
        
        # 加上相机位置得到世界坐标
        world_x = camera_pose.position.x_val + point_world_rotated[0]
        world_y = camera_pose.position.y_val + point_world_rotated[1]
        world_z = camera_pose.position.z_val + point_world_rotated[2]
        
        # 计算与相机的欧氏距离
        distance = np.linalg.norm(point_world_rotated)
        
        # 应用位置平滑
        current_pos = np.array([world_x, world_y, world_z])
        smoothed_pos = self._smooth_position(current_pos)
        
        return tuple(smoothed_pos), (pixel_x, pixel_y), depth_value, distance
    
    def _smooth_position(self, current_pos):
        """使用移动平均平滑位置数据"""
        self.position_history.append(current_pos)
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        
        weights = np.linspace(0.5, 1.0, len(self.position_history))
        weights = weights / weights.sum()
        
        smoothed = np.zeros(3)
        for i, pos in enumerate(self.position_history):
            smoothed += pos * weights[i]
        
        return smoothed
    
    def calculate_2d_ground_position(self, world_coords, camera_info):
        """计算目标在地面上的二维坐标（X,Y）"""
        world_x, world_y, world_z = world_coords
        ground_x = world_x  # 北方向坐标
        ground_y = world_y  # 东方向坐标
        return ground_x, ground_y

# 无人机控制器类
class DroneController:
    def __init__(self, client, max_speed=0.5, target_threshold=2.0, center_threshold=0.1):
        self.client = client
        self.max_speed = max_speed
        self.target_threshold = target_threshold
        self.center_threshold = center_threshold
        self.last_control_time = time.time()
        self.control_interval = 0.08
        self.target_height = args.takeoff_height
        self.velocity_smoother = VelocitySmoother(smooth_factor=0.6)
        self.last_control_cmd = {'vx': 0, 'vy': 0, 'yaw_rate': 0}
        # 添加避障控制器
        self.obstacle_avoidance = ObstacleAvoidance(self, client)
        self.avoidance_mode = False
        
    def calculate_control_command(self, target_ground_2d, image_center, bbox_center, image_size, camera_info):
        """计算控制命令使无人机飞向目标并保持目标在视野中心"""
        current_time = time.time()
        if current_time - self.last_control_time < self.control_interval:
            return self.last_control_cmd
            
        self.last_control_time = current_time
        
        try:
            drone_state = self.client.getMultirotorState()
            current_pos = drone_state.kinematics_estimated.position
            
            current_north = current_pos.x_val
            current_east = current_pos.y_val
            
            target_north, target_east = target_ground_2d
            
            distance_north = target_north - current_north
            distance_east = target_east - current_east
            distance_2d = math.sqrt(distance_north**2 + distance_east**2)
            
            target_yaw = math.atan2(distance_east, distance_north)
            current_yaw = self.get_current_yaw(drone_state)
            
            yaw_error = self.normalize_angle(target_yaw - current_yaw)
            
            img_center_x, img_center_y = image_center
            bbox_center_x, bbox_center_y = bbox_center
            img_width, img_height = image_size
            
            center_error_x = (bbox_center_x - img_center_x) / img_width
            center_error_y = (bbox_center_y - img_center_y) / img_height
            
            control_cmd = {}
            
            if distance_2d > self.target_threshold:
                speed = min(self.max_speed, distance_2d * 0.3)
                vx = max(0.1, math.cos(yaw_error)) * speed
                vy = math.sin(yaw_error) * speed * 0.8
                
                if math.cos(yaw_error) < 0:
                    vx = 0.1
                    yaw_rate = -np.sign(yaw_error) * 0.8
                else:
                    yaw_rate = 0
                
                control_cmd['vx'] = vx
                control_cmd['vy'] = vy
                control_cmd['yaw_rate'] = yaw_rate
            else:
                speed = min(self.max_speed * 0.2, distance_2d * 0.2)
                vx = max(0.05, math.cos(yaw_error)) * speed
                vy = math.sin(yaw_error) * speed * 0.5
                
                control_cmd['vx'] = vx
                control_cmd['vy'] = vy
                control_cmd['yaw_rate'] = 0
            
            # 偏航控制
            if abs(center_error_x) > self.center_threshold:
                yaw_rate = -center_error_x * 1.5
                yaw_rate = max(min(yaw_rate, 0.8), -0.8)
                control_cmd['yaw_rate'] = control_cmd.get('yaw_rate', 0) + yaw_rate
            
            # 应用速度平滑
            vx_smooth, vy_smooth, yaw_smooth = self.velocity_smoother.smooth_velocity(
                control_cmd.get('vx', 0), 
                control_cmd.get('vy', 0), 
                control_cmd.get('yaw_rate', 0)
            )
            
            control_cmd['vx'] = vx_smooth
            control_cmd['vy'] = vy_smooth  
            control_cmd['yaw_rate'] = yaw_smooth
            
            self.last_control_cmd = control_cmd
            
            return control_cmd
            
        except Exception as e:
            print(f"控制计算错误: {str(e)}")
            return self.last_control_cmd
    
    def get_current_yaw(self, drone_state):
        """从无人机状态获取当前偏航角"""
        orientation = drone_state.kinematics_estimated.orientation
        w, x, y, z = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw
    
    def normalize_angle(self, angle):
        """将角度归一化到 [-π, π] 范围"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def execute_control(self, control_cmd):
        """执行控制命令"""
        if not control_cmd:
            return
        
        try:
            self.client.moveByVelocityZAsync(
                control_cmd.get('vx', 0),
                control_cmd.get('vy', 0),
                -self.target_height,
                0.7,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(True, control_cmd.get('yaw_rate', 0))
            )
        except Exception as e:
            print(f"控制执行错误: {str(e)}")

def call_llava_for_roi_simple(frame, bbox, llava_script="llava.py"):
    """简化版本，只用于测试"""
    try:
        x, y, w, h = map(int, bbox)
        
        # 截取ROI
        roi_image = frame[y:y+h, x:x+w]
        if roi_image.size == 0:
            return "物体"
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            temp_path = tmp_file.name
            cv2.imwrite(temp_path, roi_image)
        
        # 调用llava
        result = subprocess.run(
            [sys.executable, llava_script, temp_path, '--simple'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=120
        )
        
        # 清理临时文件
        import os
        os.unlink(temp_path)
        
        if result.returncode == 0:
            output = result.stdout.strip()
            print(f"llava输出: {output}")
            
            # 简单提取：取最后一行非空行
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            if lines:
                return lines[-1]
            else:
                return "物体"
        else:
            print(f"llava错误: {result.stderr[:100]}")
            return "物体"
            
    except Exception as e:
        print(f"调用出错: {e}")
        return "物体"

def initialize_depth_model():
    """初始化深度估计模型"""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device} 进行深度估计")
        
        from depth_anything_v2.dpt import DepthAnythingV2
        
        if device == 'cuda':
            try:
                free_memory = torch.cuda.mem_get_info()[0] / 1024**3
                print(f"可用GPU显存: {free_memory:.2f} GB")
                
                if free_memory < 2.0:
                    print("显存较少，使用小型模型")
                    encoder = 'vits'
                elif free_memory < 4.0:
                    print("显存中等，使用中型模型")
                    encoder = 'vitb'
                else:
                    print("显存充足，使用大型模型")
                    encoder = args.depth_model
            except:
                encoder = 'vits'
        else:
            encoder = 'vits'
        
        print(f"选择模型: {encoder}")
        
        model = DepthAnythingV2(encoder=encoder, features=128, 
                               out_channels=[96, 192, 384, 768]).to(device).eval()
        
        if os.path.exists(args.depth_checkpoint):
            checkpoint = torch.load(args.depth_checkpoint, map_location=device)
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            print(f"深度模型加载成功: {args.depth_checkpoint}")
            
            if 'cuda' in str(device):
                try:
                    free_memory = torch.cuda.mem_get_info()[0] / 1024**3
                    if free_memory > 2.0:
                        model = model.half()
                        print("启用半精度(FP16)加速")
                    else:
                        print("显存较少，使用全精度(FP32)")
                except:
                    print("无法检测显存，使用全精度(FP32)")
        else:
            print(f"警告: 深度模型检查点不存在: {args.depth_checkpoint}")
            return None
            
        transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return DepthEstimator(model, transform, device)
        
    except Exception as e:
        print(f"深度模型初始化失败: {str(e)}")
        print("请确保已安装 depth_anything_v2 并下载了模型检查点")
        return None

# 初始化深度模型
depth_estimator = initialize_depth_model()

# 初始化跟踪器
try:
    tracker_param_path = os.path.join(args.tracker_param)
    param_module = importlib.import_module(f'lib.test.parameter.{args.tracker_name}')
    params = param_module.parameters(tracker_param_path)
    
    if not hasattr(params, 'debug'):
        params.debug = False
    if not hasattr(params, 'save_all_boxes'):
        params.save_all_boxes = False
    
    tracker_class = importlib.import_module(f'lib.test.tracker.{args.tracker_name}').get_tracker_class()
    tracker = tracker_class(params, dataset_name='vot')
except Exception as e:
    print(f"跟踪器初始化失败: {str(e)}")
    exit()

# 连接AirSim
print("正在连接AirSim...")
try:
    client = airsim.MultirotorClient(ip=args.airsim_host)
    client.confirmConnection()
    
    camera_info = client.simGetCameraInfo(args.camera_name)
    print(f"成功连接到AirSim，使用相机: {args.camera_name}")
    print(f"相机位置: X={camera_info.pose.position.x_val:.2f}, Y={camera_info.pose.position.y_val:.2f}, Z={camera_info.pose.position.z_val:.2f}")
    
except Exception as e:
    print(f"连接AirSim失败: {str(e)}")
    print("请确保AirSim正在运行且设置正确")
    exit()

# 初始化无人机控制器
drone_controller = DroneController(
    client=client,
    max_speed=args.max_speed,
    target_threshold=args.target_threshold,
    center_threshold=args.center_threshold
)

# 无人机起飞函数
def takeoff_drone(height=2.0):
    """控制无人机起飞到指定高度"""
    try:
        print(f"正在起飞到 {height} 米高度...")
        client.enableApiControl(True)
        client.armDisarm(True)
        
        takeoff_task = client.takeoffAsync()
        takeoff_task.join()
        time.sleep(2)
        
        move_task = client.moveToZAsync(-height, 1.5)
        move_task.join()
        time.sleep(1)
        
        drone_state = client.getMultirotorState()
        current_height = -drone_state.kinematics_estimated.position.z_val
        print(f"起飞完成! 当前高度: {current_height:.2f} 米")
        
        return True
        
    except Exception as e:
        print(f"起飞失败: {str(e)}")
        return False

# 执行起飞
if not takeoff_drone(args.takeoff_height):
    print("无人机起飞失败，程序退出")
    exit()

print("确保高度稳定...")
time.sleep(2)
client.moveToZAsync(-args.takeoff_height, 0.5).join()
print("高度控制已设置")

# 获取图像分辨率
responses = client.simGetImages([airsim.ImageRequest(args.camera_name, airsim.ImageType.Scene, False, False)])
if responses:
    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    if img1d.size > 0:
        img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
        actual_height, actual_width = img_rgb.shape[:2]
    else:
        actual_width, actual_height = 640, 480
else:
    actual_width, actual_height = 1920, 1080

print(f"图像分辨率: {actual_width} x {actual_height}")

# 初始化位置解算器
position_solver = PositionSolver(args.fov_degrees, actual_width, actual_height)

# 创建输出视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output_video, fourcc, 30, (actual_width, actual_height))

def select_roi(frame):
    """处理ROI选择，支持取消操作"""
    cv2.namedWindow("Select ROI (ENTER-confirm, C-cancel)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select ROI (ENTER-confirm, C-cancel)", actual_width, actual_height)
    
    roi = cv2.selectROI("Select ROI (ENTER-confirm, C-cancel)", frame, fromCenter=False)
    cv2.destroyWindow("Select ROI (ENTER-confirm, C-cancel)")
    
    if roi == (0, 0, 0, 0):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            return None
    
    return roi

# 获取第一帧
print("获取第一帧...")
responses = client.simGetImages([airsim.ImageRequest(args.camera_name, airsim.ImageType.Scene, False, False)])
if not responses or not responses[0].image_data_uint8:
    print("无法从AirSim获取图像")
    exit()

img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
frame_original = img1d.reshape(responses[0].height, responses[0].width, 3)
frame = frame_original.copy()

# 根据模式初始化跟踪信息
init_info = {}

# ROI选择部分
if args.mode in ['BBOX', 'BBOXNL']:
    while True:
        roi = select_roi(frame)
        if roi is None:
            print("已取消ROI选择，程序退出")
            exit()
        elif roi == (0, 0, 0, 0):
            print("请重新选择有效的ROI区域")
            continue
        else:
            init_info['init_bbox'] = [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]
            
            if args.mode in ['NL', 'BBOXNL']:
                print("正在使用llava分析选定目标...")
                
                if args.llava_mode == 'target_only':
                    mode = LLavaMode.TARGET_ONLY
                elif args.llava_mode == 'target_with_bbox':
                    mode = LLavaMode.TARGET_WITH_BBOX
                elif args.llava_mode == 'target_with_scene':
                    mode = LLavaMode.TARGET_WITH_SCENE
                else:
                    mode = LLavaMode.TARGET_ONLY
                
                description = call_llava_for_roi_simple(frame, roi, args.llava_path)
                print(f"llava分析结果: {description}")
                
                args.language = description
                init_info['language'] = description
                
                print(f"已设置语言跟踪目标为: {description}")
                
                temp_frame = frame.copy()
                cv2.putText(temp_frame, f"Target: {description}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
                cv2.imshow("Tracking - AirSim", temp_frame)
                cv2.waitKey(2000)
            break

if args.mode == 'NL' and not init_info.get('language'):
    while True:
        roi = select_roi(frame)
        if roi is None:
            print("已取消ROI选择，程序退出")
            exit()
        elif roi == (0, 0, 0, 0):
            print("请重新选择有效的ROI区域")
            continue
        else:
            print("正在使用llava分析选定目标...")
            
            if args.llava_mode == 'target_only':
                mode = LLavaMode.TARGET_ONLY
            elif args.llava_mode == 'target_with_bbox':
                mode = LLavaMode.TARGET_WITH_BBOX
            elif args.llava_mode == 'target_with_scene':
                mode = LLavaMode.TARGET_WITH_SCENE
            else:
                mode = LLavaMode.TARGET_ONLY
            
            description = call_llava_for_roi_simple(frame, roi, args.llava_path)
            print(f"llava分析结果: {description}")
            
            args.language = description
            init_info['language'] = description
            
            print(f"已设置语言跟踪目标为: {description}")
            
            temp_frame = frame.copy()
            cv2.putText(temp_frame, f"Target: {description}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
            cv2.imshow("Tracking - AirSim", temp_frame)
            cv2.waitKey(2000)
            break

# 初始化跟踪器
try:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tracker.initialize(frame_rgb, init_info)
    print("跟踪器初始化成功")
except Exception as e:
    print(f"跟踪器初始化失败: {str(e)}")
    exit()

# 创建显示窗口
cv2.namedWindow("Tracking - AirSim", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracking - AirSim", actual_width, actual_height)

# 主处理循环
frame_count = 0
total_time = 0
paused = False

print("\n操作说明:")
print("按 'q' 键退出")
print("按 'p' 键暂停/继续")
print("按 's' 键重新选择ROI")
print("按 'c' 键切换控制开关")
print("按 'r' 键重置无人机位置")

def reset_drone():
    """重置无人机到初始位置"""
    try:
        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()
        client.moveToZAsync(-args.takeoff_height, 2).join()
        print("无人机已重置")
    except Exception as e:
        print(f"重置无人机失败: {str(e)}")

try:
    while True:
        if not paused:
            responses = client.simGetImages([airsim.ImageRequest(args.camera_name, airsim.ImageType.Scene, False, False)])
            if not responses or not responses[0].image_data_uint8:
                print("无法从AirSim获取图像")
                break
            
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            frame_original = img1d.reshape(responses[0].height, responses[0].width, 3)
            frame = frame_original.copy()
            
            start_time = time.time()
            
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                output = tracker.track(frame_rgb, {})
                bbox = output['target_bbox']
                x, y, w, h = map(int, bbox)
                
                # 估计深度
                depth_gray = depth_estimator.estimate_depth(frame_rgb) if depth_estimator else None
                depth_for_calculation = depth_estimator.get_raw_depth_for_calculation(frame_rgb) if depth_estimator else None
                
                # 计算世界坐标和二维地面坐标
                world_coords = None
                ground_2d = None
                distance = 0
                if depth_for_calculation is not None:
                    world_coords, center_pixel, depth_value, distance = position_solver.calculate_world_coordinates(
                        [x, y, w, h], depth_for_calculation, camera_info)
                    
                    if world_coords:
                        ground_2d = position_solver.calculate_2d_ground_position(world_coords, camera_info)
                
                # 在可写的帧上绘制边界框
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 绘制中心点和底部中心点
                center_x = x + w // 2
                center_y = y + h // 2
                bottom_center_y = y + h
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.circle(frame, (center_x, bottom_center_y), 5, (255, 0, 0), -1)
                
                # 在左上角显示灰度深度图
                if depth_gray is not None:
                    depth_height, depth_width = actual_height // 4, actual_width // 4
                    depth_resized = cv2.resize(depth_gray, (depth_width, depth_height))
                    depth_bgr = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)
                    frame[10:10+depth_height, 10:10+depth_width] = depth_bgr
                    
                    cv2.rectangle(frame, (8, 8), (12+depth_width, 12+depth_height), (255, 255, 255), 2)
                    cv2.putText(frame, "Depth Gray", (15, 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 计算FPS
                fps = 1/(time.time()-start_time) if (time.time()-start_time) > 0 else 0
                
                # 显示信息
                y_pos = 30
                info = [
                    f"Language: {args.language}",
                    f"FPS: {fps:.1f}"
                ]
                
                # 控制逻辑
                control_info = []  # 初始化control_info
                
                if args.control_enabled and depth_for_calculation is not None:
                    depth_for_avoidance = depth_for_calculation.astype(np.float32)
                    
                    dt = time.time() - getattr(drone_controller, 'last_control_time', time.time())
                    avoidance_active = drone_controller.obstacle_avoidance.execute_avoidance(depth_for_avoidance, dt)
                    
                    if avoidance_active:
                        drone_controller.avoidance_mode = True
                        control_info = [
                            "Mode: AVOIDANCE",
                            "Avoiding obstacles"
                        ]
                        
                        for i, text in enumerate(control_info):
                            cv2.putText(frame, text, (10, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            y_pos += 25
                        
                        drone_controller.obstacle_avoidance.update_trajectory()
                    else:
                        drone_controller.avoidance_mode = False
                        
                        if world_coords and ground_2d:
                            image_center = (actual_width // 2, actual_height // 2)
                            bbox_center = (x + w // 2, y + h // 2)
                            image_size = (actual_width, actual_height)
                            
                            control_cmd = drone_controller.calculate_control_command(
                                ground_2d, image_center, bbox_center, image_size, camera_info
                            )
                            
                            if control_cmd:
                                drone_controller.execute_control(control_cmd)
                                
                                control_info = [
                                    f"Control: vx={control_cmd.get('vx', 0):.2f}, vy={control_cmd.get('vy', 0):.2f}",
                                    f"Yaw Rate: {control_cmd.get('yaw_rate', 0):.2f}"
                                ]
                        
                        # 显示控制信息
                        for i, text in enumerate(control_info):
                            cv2.putText(frame, text, (10, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            y_pos += 25
                        
                        drone_controller.obstacle_avoidance.update_trajectory()
                else:
                    cv2.putText(frame, "Control: OFF", (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    y_pos += 25
                
                # 显示其他信息
                info_start_y = max(y_pos, actual_height // 4 + 40)
                for i, text in enumerate(info):
                    color = (0, 0, 255)  # 红色
                    cv2.putText(frame, text, (10, info_start_y + i * 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            except Exception as e:
                print(f"跟踪出错: {str(e)}")
                import traceback
                traceback.print_exc()
                break
            
            # 写入输出
            out.write(frame)
            
            frame_count += 1
            total_time += time.time() - start_time
        
        # 显示
        cv2.imshow("Tracking - AirSim", frame)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("已暂停" if paused else "已继续")
        elif key == ord('s'):
            if args.mode in ['BBOX', 'BBOXNL']:
                new_roi = select_roi(frame)
                if new_roi is not None and new_roi != (0, 0, 0, 0):
                    init_info['init_bbox'] = [int(new_roi[0]), int(new_roi[1]), int(new_roi[2]), int(new_roi[3])]
                    if args.mode in ['NL', 'BBOXNL']:
                        init_info['language'] = args.language
                    tracker.initialize(frame_rgb, init_info)
                    print("重新初始化跟踪器")
        elif key == ord('c'):
            args.control_enabled = not args.control_enabled
            status = "启用" if args.control_enabled else "禁用"
            print(f"无人机控制已{status}")
        elif key == ord('r'):
            reset_drone()

except KeyboardInterrupt:
    print("\n程序被用户中断")

finally:
    # 清理
    print("正在清理...")
    cv2.destroyAllWindows()
    out.release()
    
    # 清除轨迹
    try:
        client.simFlushPersistentMarkers()
    except:
        pass
    
    # 安全降落
    try:
        print("正在降落...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
    except:
        pass
    
    print(f"处理完成: {frame_count} 帧, 平均FPS: {frame_count/total_time:.2f}")
