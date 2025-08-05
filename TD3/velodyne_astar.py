import math
import os
import random
import subprocess
import time
from os import path
import csv
import numpy as np
import rospy

import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from astar import AStar
from utils import Grid
from collision_utils import check_collision

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1



# 检查(x, y)这个点是否在障碍物区域内，如果在障碍物上则返回False，否则返回True
def check_pos(x, y):
    goal_ok = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok


class GazeboEnv:
    def visualize_obstacles(self):
        obstacles_rect = [
    (-3.43, 1.76, 2.59, 0.21),  # 11*
    (-2.28, 0.35, 0.21, 2.84),   # 13*
    (2.06, 2.93, 1.34, 0.21),  # 15*
    (2.06, 2.11, 0.21, 1.04),  # 16*
    (2.09, 2.11, 1.34, 0.21),  # 17*
    (3.22, 2.11, 0.21, 1.04), # 18*

    (-3.80, -3.39, 2.04, 0.21), # 24
    (-3.00, -3.39, 2.14, 0.21), # 25
    (-3.42, -3.39, 2.09, 0.21), # 26

    (1.48, -2.59, 1.84, 0.24), # 28*
    (3.10, -2.57, 0.24, 2.59), # 29*
    (-5.52, 5.33, 11.02, 0.17), # 6*
    (-5.52, -5.52, 0.17, 11.02), # 7*
    (-5.52, -5.52, 11.02, 0.17), # 8*
    (5.33, -5.52, 0.17, 11.02), # 9*

    (-4.61, 4.52, 1.09, 1.09), # fire_hydrant
    (4.30, -3.77, 0.99, 0.10), # back
    (5.19, -4.16, 0.11, 0.49),  # left_side
    (4.29, -4.16, 0.11, 0.49), # right_side

    (4.31, -4.16, 0.97, 0.49), # bottom
    (4.31, -4.16, 0.97, 0.49), # top
    (4.31, -4.16, 0.97, 0.49), # low_shelf
    (4.31, -4.16, 0.97, 0.49), # high_shelf

    # (-5.11, -0.96, 1.57, 0.87), # surface
    (-3.72, -0.21, 0.13, 0.13), # front_left_leg*
    (-3.72, -0.97, 0.13, 0.13), # front_right_leg*
    (-5.07, -0.97, 0.13, 0.13), # back_right_leg*
    (-5.07, -0.21, 0.13, 0.13), # back_left_leg*

    (3.68, 0.74, 0.59, 0.49), # cardboard_box_0*
    (-0.28, -4.20, 0.59, 0.49), # cardboard_box_1*
    (-5.17, 2.74, 0.59, 0.49), # cardboard_box_2*
    (-0.30, 3.76, 0.59, 0.49), # cardboard_box_3*
    ]
        
        markerArray = MarkerArray()
        for i, (ox, oy, length, width) in enumerate(obstacles_rect):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.scale.x = length
            marker.scale.y = width
            marker.scale.z = 0.1
            marker.color.a = 0.7
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = ox + length / 2
            marker.pose.position.y = oy + width / 2
            marker.pose.position.z = 0.05
            marker.id = i + 100000  # 防止与其他marker冲突
            markerArray.markers.append(marker)
        self.publisher.publish(markerArray)

    
    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim # 激光雷达数据维度
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 5.0 # 目标点随机生成的上界
        self.lower = -5.0 # 目标点随机生成的下界
        self.velodyne_data = np.ones(self.environment_dim) * 10 # 初始化激光雷达数据，每个扇区的距离都设为10米（表示很远，没有障碍物）
        self.last_odom = None # 最近一次里程计数据

        self.path = [[self.goal_x, self.goal_y]]

        # 初始化机器人模型状态
        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        # 计算每个激光扇区的角度范围
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03


        port = "11311"
        subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        try:
            rospy.init_node("gym", anonymous=True)
        except rospy.exceptions.ROSException:
            # 如果节点已经初始化，继续执行
            pass
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3) # 目标点可视化
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1) # 线速度可视化
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1) # 角速度可视化
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        ) # 激光点云
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        ) # 里程计

    # 将点云数据分成多个扇区，每个扇区只保留最小距离（距离最近的障碍物的距离），作为状态输入
    def velodyne_callback(self, v):
        # 处理激光点云数据，将其转为每个扇区的最小距离
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2: # 只考虑地面以上的点
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def odom_callback(self, od_data):
        self.last_odom = od_data # 保存最新的里程计数据



    def step(self, action):
        target = False # 标记是否到达目标点，初始为False
        local_target = False # 标记是否到达局部目标点，初始为False

        # 1. 发布机器人动作
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")


        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # 读取激光雷达数据，判断是否碰撞
        done, collision, min_laser = self.observe_collision(self.velodyne_data) # 判断是否终止、是否碰撞、最小激光距离
        v_state = []
        v_state[:] = self.velodyne_data[:] # 复制当前激光数据
        laser_state = [v_state] # 包装成列表，便于后续拼接

        # 读取机器人当前位置和朝向
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        if self.path:
            first_local_x, first_local_y = self.path[-1]
            local_goal_distance = np.linalg.norm(
                [first_local_x - self.odom_x, first_local_y - self.odom_y]
            )
            if local_goal_distance < GOAL_REACHED_DIST:
                local_target = True
                self.path.pop()
        else:
            # fallback: 没有局部目标点，直接用当前位置
            first_local_x, first_local_y = self.odom_x, self.odom_y
            local_goal_distance = 0.0

        # 计算机器人朝向与目标方向的夹角
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        if distance < GOAL_REACHED_DIST:
            target = True
            done = True

        robot_state = [distance, theta, action[0], action[1], local_goal_distance] # 机器人状态：距离、角度、线速度、角速度
        state = np.append(laser_state, robot_state) # 拼接激光数据和机器人状态，作为新状态
        reward = self.get_reward(target, collision, action, min_laser, local_target) # 计算奖励
        return state, reward, done, target

    def reset(self):

        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = round(np.random.uniform(-4.5, 4.5), 2)
            y = round(np.random.uniform(-4.5, 4.5), 2)
            position_ok = check_pos(x, y) and check_collision(x, y)
        print(f"随机生成起点: ({x:.2f}, {y:.2f})")
        object_state.pose.position.x = x
        object_state.pose.position.y = y


        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y


        self.change_goal()

        obstacles_rect = [
    (-3.43, 1.76, 2.59, 0.21),  # 11*
    (-2.28, 0.35, 0.21, 2.84),   # 13*
    (2.06, 2.93, 1.34, 0.21),  # 15*
    (2.06, 2.11, 0.21, 1.04),  # 16*
    (2.09, 2.11, 1.34, 0.21),  # 17*
    (3.22, 2.11, 0.21, 1.04), # 18*

    (-3.80, -3.39, 2.04, 0.21), # 24
    (-3.00, -3.39, 2.14, 0.21), # 25
    (-3.42, -3.39, 2.09, 0.21), # 26

    (1.48, -2.59, 1.84, 0.24), # 28*
    (3.10, -2.57, 0.24, 2.59), # 29*
    (-5.52, 5.33, 11.02, 0.17), # 6*
    (-5.52, -5.52, 0.17, 11.02), # 7*
    (-5.52, -5.52, 11.02, 0.17), # 8*
    (5.33, -5.52, 0.17, 11.02), # 9*

    (-4.61, 4.52, 1.09, 1.09), # fire_hydrant
    (4.30, -3.77, 0.99, 0.10), # back
    (5.19, -4.16, 0.11, 0.49),  # left_side
    (4.29, -4.16, 0.11, 0.49), # right_side

    (4.31, -4.16, 0.97, 0.49), # bottom
    (4.31, -4.16, 0.97, 0.49), # top
    (4.31, -4.16, 0.97, 0.49), # low_shelf
    (4.31, -4.16, 0.97, 0.49), # high_shelf

    # (-5.11, -0.96, 1.57, 0.87), # surface
    (-3.72, -0.21, 0.13, 0.13), # front_left_leg*
    (-3.72, -0.97, 0.13, 0.13), # front_right_leg*
    (-5.07, -0.97, 0.13, 0.13), # back_right_leg*
    (-5.07, -0.21, 0.13, 0.13), # back_left_leg*

    (3.68, 0.74, 0.59, 0.49), # cardboard_box_0*
    (-0.28, -4.20, 0.59, 0.49), # cardboard_box_1*
    (-5.17, 2.74, 0.59, 0.49), # cardboard_box_2*
    (-0.30, 3.76, 0.59, 0.49), # cardboard_box_3*
    ]

        # 创建网格地图 
        resolution = 0.01 
        width = 1000
        height = 1000
        grid = [[0 for _ in range(height)] for _ in range(width)]
        
        # 将障碍物信息添加到网格中
        for obs in obstacles_rect:
            def world_to_grid(x, y):
                # 将世界坐标转换为网格坐标，使用round避免截断误差
                grid_x = int(round((x + 5.0) / resolution))
                grid_y = int(round((y + 5.0) / resolution))
                return grid_x, grid_y

            def world_size_to_grid(size):
                # 将世界尺寸转换为网格尺寸
                return int(round(size / resolution))

            # 将实际坐标转换为网格坐标
            x_start, y_start = world_to_grid(obs[0], obs[1])
            x_size = world_size_to_grid(obs[2])
            y_size = world_size_to_grid(obs[3])
            x_end = x_start + x_size
            y_end = y_start + y_size
            
            # 确保在网格范围内
            x_start = max(0, min(x_start, width))
            x_end = max(0, min(x_end, width))
            y_start = max(0, min(y_start, height))
            y_end = max(0, min(y_end, height))
            
            # 标记障碍物
            for x in range(x_start, x_end):
                for y in range(y_start, y_end):
                    if 0 <= x < width and 0 <= y < height:
                        grid[x][y] = 1

        # 创建网格环境
        env = Grid(width, height)
        env.grid = grid

        # 打印所有障碍物网格点的x和y索引
        # print("障碍物网格点 (x, y):")
        # line = []
        # for x in range(width):
        #     for y in range(height):
        #         if grid[x][y] == 1:
        #             line.append(f"({x},{y})")
        # print(' '.join(line))
        
        # 将连续坐标转换为网格坐标
        start = world_to_grid(self.odom_x, self.odom_y)
        goal = world_to_grid(self.goal_x, self.goal_y)
        
        # 使用A*算法进行路径规划
        planner = AStar(start=start, goal=goal, env=env)
        cost, path, expand = planner.plan()
        # 打印网格路径坐标
        # if path:
        #     print("\n网格路径坐标：", end='')
        #     print(' '.join([f"({grid_x},{grid_y})" for grid_x, grid_y in path]))
        
        # print(f"\nA*路径规划：")
        # print(f"起点: ({self.odom_x:.2f}, {self.odom_y:.2f})")
        # print(f"终点: ({self.goal_x:.2f}, {self.goal_y:.2f})")
        
        if path:
            print("\n路径点坐标：")
            for grid_x, grid_y in path:
                # 从网格坐标转换回实际坐标
                real_x = grid_x * resolution - 5.0
                real_y = grid_y * resolution - 5.0
                print(f"({real_x:.2f}, {real_y:.2f})")
        
        if path:
            def grid_to_world(grid_x, grid_y):
                # 从网格坐标转换回世界坐标，保持精确性
                world_x = grid_x * resolution - 5.0
                world_y = grid_y * resolution - 5.0
                # 确保坐标在有效范围内
                world_x = max(-5.0, min(5.0, world_x))
                world_y = max(-5.0, min(5.0, world_y))
                return world_x, world_y

            # 将网格坐标转换回连续坐标
            continuous_path = []
            for grid_x, grid_y in path:
                world_x, world_y = grid_to_world(grid_x, grid_y)
                continuous_path.append([world_x, world_y])
            self.path = continuous_path
        else:
            print("A* path not found, using goal as path")
            self.path = [[self.goal_x, self.goal_y], [self.odom_x, self.odom_y]]

        self.publish_path(self.path)

        # 确保物理引擎运行以获取准确的雷达数据
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # 收集初始雷达数据
        rospy.sleep(0.5)  # 等待雷达数据更新
        initial_laser_data = self.velodyne_data[:]


        # === 角度控制旋转180°，先快后慢 ===
        vel_cmd = Twist()
        # 记录初始角度
        if self.last_odom is not None:
            q = self.last_odom.pose.pose.orientation
            quat = Quaternion(q.w, q.x, q.y, q.z)
            euler = quat.to_euler(degrees=False)
            start_yaw = euler[2]
        else:
            start_yaw = 0.0
        rotated = 0.0
        prev_yaw = start_yaw
        rate = rospy.Rate(50)
        while abs(rotated) < np.pi:
            # 先快后慢，剩余小于0.3弧度时减速
            remain = np.pi - abs(rotated)
            if remain > 0.3:
                vel_cmd.angular.z = 1.0
            elif remain > 0.1:
                vel_cmd.angular.z = 0.3
            else:
                vel_cmd.angular.z = 0.05
            self.vel_pub.publish(vel_cmd)
            self.publish_markers([0.0, vel_cmd.angular.z])
            if self.last_odom is not None:
                q = self.last_odom.pose.pose.orientation
                quat = Quaternion(q.w, q.x, q.y, q.z)
                euler = quat.to_euler(degrees=False)
                curr_yaw = euler[2]
            else:
                curr_yaw = prev_yaw
            delta_yaw = curr_yaw - prev_yaw
            # print(f"当前角度: {curr_yaw:.3f}, 之前角度: {prev_yaw:.3f}, 旋转增量: {delta_yaw:.3f}, 累计旋转: {rotated:.3f}")
            if delta_yaw > np.pi:
                delta_yaw -= 2 * np.pi
            elif delta_yaw < -np.pi:
                delta_yaw += 2 * np.pi
            rotated += delta_yaw
            prev_yaw = curr_yaw
            rate.sleep()
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)
        rospy.sleep(0.5)

        # 收集旋转180°后的雷达数据
        rotated_laser_data = self.velodyne_data[:]

        optimal_subgoal = self.find_optimal_subgoal()
        self.publish_path(self.path, subgoal_point=optimal_subgoal)
        # 打印最优子目标点坐标
        if optimal_subgoal is not None:
            print(f"最优子目标点坐标: ({optimal_subgoal[0]:.3f}, {optimal_subgoal[1]:.3f})\n")
        else:
            print("未找到最优子目标点\n")

        # 准备写入数据，插入最优子目标点坐标
        row_data = [self.odom_x, self.odom_y, self.goal_x, self.goal_y]
        # 插入最优子目标点坐标
        if optimal_subgoal is not None:
            row_data.extend([optimal_subgoal[0], optimal_subgoal[1]])
        else:
            row_data.extend([None, None])
        row_data.extend(initial_laser_data)  # 初始雷达数据
        row_data.extend(rotated_laser_data)  # 旋转后雷达数据

        # 写入前检查是否已收集3000行数据（含表头共3001行）
        csv_path = 'new_laser_data.csv'
        try:
            with open(csv_path, 'r') as f:
                row_count = sum(1 for _ in f)
        except FileNotFoundError:
            row_count = 0
        # 第一行为表头，实际数据行数=row_count-1
        if row_count >= 3001:
            print("已收集3000行数据，停止写入new_laser_data.csv。");
            return
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_data)

        # === 角度控制旋转回原位，先快后慢 ===
        # 记录当前角度为回转起点
        if self.last_odom is not None:
            q = self.last_odom.pose.pose.orientation
            quat = Quaternion(q.w, q.x, q.y, q.z)
            euler = quat.to_euler(degrees=False)
            start_yaw = euler[2]
        else:
            start_yaw = 0.0
        rotated = 0.0
        prev_yaw = start_yaw
        rate = rospy.Rate(50)
        while abs(rotated) < np.pi:
            remain = np.pi - abs(rotated)
            if remain > 0.3:
                vel_cmd.angular.z = -1.0
            elif remain > 0.1:
                vel_cmd.angular.z = -0.3
            else:
                vel_cmd.angular.z = -0.1
            self.vel_pub.publish(vel_cmd)
            self.publish_markers([0.0, vel_cmd.angular.z])
            if self.last_odom is not None:
                q = self.last_odom.pose.pose.orientation
                quat = Quaternion(q.w, q.x, q.y, q.z)
                euler = quat.to_euler(degrees=False)
                curr_yaw = euler[2]
            else:
                curr_yaw = prev_yaw
            delta_yaw = curr_yaw - prev_yaw
            if delta_yaw > np.pi:
                delta_yaw -= 2 * np.pi
            elif delta_yaw < -np.pi:
                delta_yaw += 2 * np.pi
            rotated += delta_yaw
            prev_yaw = curr_yaw
            rate.sleep()
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)
        rospy.sleep(0.5)

        # 暂停物理引擎
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        self.random_box()
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        if self.path:
            first_local_x, first_local_y = self.path[-1]
            local_goal_distance = np.linalg.norm(
                [first_local_x - self.odom_x, first_local_y - self.odom_y]
            )
        else:
            local_goal_distance = distance

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0, local_goal_distance]
        state = np.append(laser_state, robot_state)
        self.visualize_obstacles()
        return state

    def find_optimal_subgoal(self):
        """
        返回路径上倒数第二个点（即起点的下一个点）作为最优子目标点。
        如果路径点不足2个，则返回None。
        """
        if self.path and len(self.path) >= 2:
            return tuple(self.path[-2])
        return None

    def change_goal(self):
        # 随机生成一个新的目标点，并确保它不会出现在障碍物上或地图外
        # 同时随着训练进行，目标点的随机范围会逐渐扩大，让任务更有挑战性
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False
        while not goal_ok:
            self.goal_x = round(self.odom_x + random.uniform(self.upper, self.lower), 2)
            self.goal_y = round(self.odom_y + random.uniform(self.upper, self.lower), 2)
            goal_ok = check_pos(self.goal_x, self.goal_y) and check_collision(self.goal_x, self.goal_y)
        print(f"随机生成终点: ({self.goal_x:.2f}, {self.goal_y:.2f})", end="")

    def random_box(self):
        for i in range(0):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    def publish_path(self, path, subgoal_point=None):
        """
        发布路径点，subgoal_point为最优子目标点坐标（高亮显示），并用浅蓝色实线连接所有路径点
        """
        markerArray = MarkerArray()
        # 1. 路径点可视化（球）
        for i, point in enumerate(path):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.05
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            # 普通路径点（蓝色）
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = 0
            marker.id = i
            markerArray.markers.append(marker)

        # 2. 路径线可视化（浅蓝色实线）
        if len(path) >= 2:
            line_marker = Marker()
            line_marker.header.frame_id = "odom"
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.03  # 线宽
            line_marker.color.a = 1.0
            line_marker.color.r = 0.3
            line_marker.color.g = 0.7
            line_marker.color.b = 1.0  # 浅蓝色
            line_marker.pose.orientation.w = 1.0
            line_marker.id = 9999  # 不与点id冲突
            line_marker.points = []
            from geometry_msgs.msg import Point
            for pt in path:
                p = Point()
                p.x = pt[0]
                p.y = pt[1]
                p.z = 0
                line_marker.points.append(p)
            markerArray.markers.append(line_marker)

        # 3. 最优子目标点高亮（红色球）
        if subgoal_point is not None:
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.08
            marker.scale.y = 0.13
            marker.scale.z = 0.13
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = subgoal_point[0]
            marker.pose.position.y = subgoal_point[1]
            marker.pose.position.z = 0
            marker.id = 10000  # 不与路径点冲突
            markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser, local_target=False):
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            if local_target:
                r3 = lambda x: 1 - x if x < 1 else 0.0
                return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2 + 1
            else:
                r3 = lambda x: 1 - x if x < 1 else 0.0
                return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2