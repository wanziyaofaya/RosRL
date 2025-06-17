import math
import random
import time
import numpy as np
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from squaternion import Quaternion

from velodyne_env import GazeboEnv, TIME_DELTA, check_pos

class DynamicObstacleGazeboEnv(GazeboEnv):
    """扩展的Gazebo环境，包含动态障碍物"""

    def __init__(self, launchfile, environment_dim):
        super().__init__(launchfile, environment_dim)
        
        # 添加动态障碍物的属性
        self.dynamic_obstacles = []
        self.num_dynamic_obstacles = 3  # 设置动态障碍物的数量
        self.obstacle_speeds = []
        self.obstacle_directions = []
        
        # 初始化动态障碍物
        self.init_dynamic_obstacles()

    def init_dynamic_obstacles(self):
        """初始化动态障碍物"""
        self.dynamic_obstacles = []
        self.obstacle_speeds = []
        self.obstacle_directions = []
        
        for i in range(self.num_dynamic_obstacles):
            # 随机生成障碍物的位置（确保不在起点和终点附近）
            while True:
                x = random.uniform(-4.0, 4.0)
                y = random.uniform(-4.0, 4.0)
                # 确保障碍物不在机器人起点和目标点附近
                if (abs(x) > 1.0 or abs(y) > 1.0) and check_pos(x, y):
                    break
            
            self.dynamic_obstacles.append([x, y])
            # 随机生成速度（0.2-0.5 m/s）
            self.obstacle_speeds.append(random.uniform(0.2, 0.5))
            # 随机生成运动方向（弧度）
            self.obstacle_directions.append(random.uniform(0, 2 * np.pi))

    def update_dynamic_obstacles(self):
        """更新动态障碍物的位置"""
        for i in range(len(self.dynamic_obstacles)):
            # 更新障碍物位置
            dx = self.obstacle_speeds[i] * np.cos(self.obstacle_directions[i]) * TIME_DELTA
            dy = self.obstacle_speeds[i] * np.sin(self.obstacle_directions[i]) * TIME_DELTA
            
            new_x = self.dynamic_obstacles[i][0] + dx
            new_y = self.dynamic_obstacles[i][1] + dy
            
            # 如果障碍物即将超出边界，改变其方向
            if new_x > 4.5 or new_x < -4.5 or new_y > 4.5 or new_y < -4.5:
                self.obstacle_directions[i] = (self.obstacle_directions[i] + np.pi) % (2 * np.pi)
                dx = self.obstacle_speeds[i] * np.cos(self.obstacle_directions[i]) * TIME_DELTA
                dy = self.obstacle_speeds[i] * np.sin(self.obstacle_directions[i]) * TIME_DELTA
                new_x = self.dynamic_obstacles[i][0] + dx
                new_y = self.dynamic_obstacles[i][1] + dy
            
            self.dynamic_obstacles[i] = [new_x, new_y]
            
            # 更新Gazebo中障碍物的位置
            obstacle_state = ModelState()
            obstacle_state.model_name = f"dynamic_obstacle_{i}"
            obstacle_state.pose.position.x = new_x
            obstacle_state.pose.position.y = new_y
            obstacle_state.pose.position.z = 0.5
            self.set_state.publish(obstacle_state)

    def step(self, action):
        # 执行原始环境的step
        state = super().step(action)
        
        # 更新动态障碍物的位置
        self.update_dynamic_obstacles()
        
        # 检查是否与动态障碍物发生碰撞
        for obstacle in self.dynamic_obstacles:
            distance = np.linalg.norm([self.odom_x - obstacle[0], self.odom_y - obstacle[1]])
            if distance < 0.35:  # 碰撞距离阈值
                return state, -200, True  # 发生碰撞，返回负奖励并结束回合
        
        return state

    def reset(self):
        # 执行原始环境的reset
        state = super().reset()
        
        # 重新初始化动态障碍物
        self.init_dynamic_obstacles()
        
        return state
