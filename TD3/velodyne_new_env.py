import math
import os
import random
import subprocess
import time
from os import path
import heapq

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
from scipy.ndimage import distance_transform_edt

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1


# 判断某个点（x,y）是否在障碍物区域或地图边界外
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

# 构建栅格地图（带障碍物和障碍物距离信息）
def build_grid_map(resolution=0.2, x_range=(-4.5, 4.5), y_range=(-4.5, 4.5)):
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_num = int((x_max - x_min) / resolution) + 1
    y_num = int((y_max - y_min) / resolution) + 1
    grid = np.zeros((x_num, y_num), dtype=np.int8)
    for i in range(x_num):
        for j in range(y_num):
            x = x_min + i * resolution
            y = y_min + j * resolution
            if not check_pos(x, y):
                grid[i, j] = 1  # 1表示障碍物
    # 新增：计算每个格子到最近障碍物的距离（米）
    obs_mask = (grid == 0).astype(np.uint8)
    obs_dist = distance_transform_edt(obs_mask) * resolution # 每个格子到最近障碍物的距离
    return grid, x_min, y_min, resolution, obs_dist


def astar(grid, start, goal, obs_dist=None, alpha=2.0):
    def heuristic(a, b):
        base = np.linalg.norm(np.array(a) - np.array(b))
        if obs_dist is not None:
            d_obs = obs_dist[a[0], a[1]]
            obs_penalty = alpha * (1.0 / (d_obs + 1e-3)) if d_obs < 1.0 else 0
        else:
            obs_penalty = 0
        return base + obs_penalty

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            return data[::-1]
        close_set.add(current)
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return []


def bresenham_line(p1, p2):
    # 返回p1到p2之间所有格子坐标（含首尾）
    x0, y0 = p1
    x1, y1 = p2
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points


def is_line_free(grid, p1, p2):
    # 检查p1到p2之间的格子是否都可通行
    for x, y in bresenham_line(p1, p2):
        if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
            return False
        if grid[x, y] == 1:
            return False
    return True


def sparsify_path(grid, path):
    if not path:
        return []
    new_path = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if is_line_free(grid, path[i], path[j]):
                break
            j -= 1
        new_path.append(path[j])
        i = j
    return new_path


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 5.0
        self.lower = -5.0
        self.velodyne_data = np.ones(self.environment_dim) * 10
        self.last_odom = None

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

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
        rospy.init_node("gym", anonymous=True)
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
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        )

        self.grid, self.x_min, self.y_min, self.grid_resolution, self.obs_dist = build_grid_map()
        self.global_path = []
        self.global_path_index = 0
        self.local_goal = (self.goal_x, self.goal_y)

    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
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
        self.last_odom = od_data

    def plan_global_path(self, start_xy, goal_xy):
        def to_grid(x, y):
            i = int((x - self.x_min) / self.grid_resolution)
            j = int((y - self.y_min) / self.grid_resolution)
            return (i, j)

        def to_world(i, j):
            x = self.x_min + i * self.grid_resolution
            y = self.y_min + j * self.grid_resolution
            return (x, y)

        start_idx = to_grid(*start_xy)
        goal_idx = to_grid(*goal_xy)
        path_idx = astar(self.grid, start_idx, goal_idx, self.obs_dist)
        # 可选：路径稀疏化，去除冗余节点
        path_idx = sparsify_path(self.grid, path_idx)
        if not path_idx:
            return []
        path_xy = [to_world(i, j) for i, j in path_idx]
        return path_xy

    # Perform an action and read a new state
    def step(self, action):
        target = False

        # Publish the robot action
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

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
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

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
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

        # 计算目标方向和机器人朝向的单位向量
        goal_direction = np.array([skew_x, skew_y]) / (np.linalg.norm([skew_x, skew_y]) + 1e-8)
        robot_heading = np.array([np.cos(angle), np.sin(angle)])

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True

        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        # 加入A*启发值
        astar_heuristic = self.compute_astar_heuristic(self.odom_x, self.odom_y, self.goal_x, self.goal_y)
        state = np.append(state, astar_heuristic)
       
        reward = self.get_reward(target, collision, action, min_laser, goal_direction, robot_heading, distance)
        # 动态切换局部目标点
        local_goal_x, local_goal_y = self.local_goal
        distance_to_local_goal = np.linalg.norm([self.odom_x - local_goal_x, self.odom_y - local_goal_y])
        if distance_to_local_goal < 0.3 and self.global_path_index < len(self.global_path) - 1:
            self.global_path_index += 1
            self.local_goal = self.global_path[self.global_path_index]
        return state, reward, done, target

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
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
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # set a random goal in empty space in environment
        self.change_goal()
        # 生成全局路径并初始化局部目标点
        self.global_path = self.plan_global_path((self.odom_x, self.odom_y), (self.goal_x, self.goal_y))
        self.global_path_index = 1 if len(self.global_path) > 1 else 0
        if self.global_path:
            self.local_goal = self.global_path[self.global_path_index]
        else:
            self.local_goal = (self.goal_x, self.goal_y)
        # randomly scatter boxes in the environment
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

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        # === 新增：A*启发值 ===
        astar_heuristic = self.compute_astar_heuristic(self.odom_x, self.odom_y, self.goal_x, self.goal_y)
        state = np.append(state, astar_heuristic)
        # === END ===
        return state

    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)
        # 重新生成全局路径和局部目标点
        self.global_path = self.plan_global_path((self.odom_x, self.odom_y), (self.goal_x, self.goal_y))
        self.global_path_index = 1 if len(self.global_path) > 1 else 0
        if self.global_path:
            self.local_goal = self.global_path[self.global_path_index]
        else:
            self.local_goal = (self.goal_x, self.goal_y)

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
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

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser, goal_direction, robot_heading, distance_to_goal):
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            # cos(theta) 是动作是否朝向目标
            cos_theta = np.dot(goal_direction, robot_heading)

            forward_reward = action[0] * max(0, cos_theta)  # 只奖励正向前进
            turn_penalty = abs(action[1])
            laser_penalty = np.exp(-min_laser)
            goal_penalty = 0.1 * distance_to_goal

            reward = forward_reward - 0.5 * turn_penalty - 0.5 * laser_penalty - goal_penalty

            # 加一条安全检查：速度太快但太近就惩罚
            if min_laser < 0.5 and action[0] > 0.5:
                reward -= 0.2

            return reward

    def compute_forward_obstacle_density(self, angle, d_thresh=10.0):
        """
        计算机器人前进方向±90°内的障碍物密度
        angle: 机器人当前朝向（弧度）
        d_thresh: 小于该距离视为有障碍物
        返回：密度（0~1）
        """
        laser = np.array(self.velodyne_data)
        num_rays = len(laser)
        # 假设velodyne_data均匀分布在[-pi, pi]
        angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False)
        # 选取前进方向±90°的激光
        mask = (angles >= angle - np.pi/2) & (angles <= angle + np.pi/2)
        selected = laser[mask]
        num_obstacle = np.sum(selected < d_thresh)
        density = num_obstacle / selected.size if selected.size > 0 else 0
        return density

    def compute_astar_heuristic(self, x, y, goal_x, goal_y):
        """
        计算当前位置到目标点的A*启发值（障碍物距离加权欧氏距离+障碍物密度因子）
        """
        # 转为栅格坐标
        i = int((x - self.x_min) / self.grid_resolution)
        j = int((y - self.y_min) / self.grid_resolution)
        gi = int((goal_x - self.x_min) / self.grid_resolution)
        gj = int((goal_y - self.y_min) / self.grid_resolution)
        # 欧氏距离
        base = np.linalg.norm([i - gi, j - gj]) * self.grid_resolution
        # 障碍物距离惩罚
        if 0 <= i < self.obs_dist.shape[0] and 0 <= j < self.obs_dist.shape[1]:
            d_obs = self.obs_dist[i, j]
            obs_penalty = 2.0 * (1.0 / (d_obs + 1e-3)) if d_obs < 1.0 else 0
        else:
            obs_penalty = 0
        # === 新增：障碍物密度因子 ===
        if hasattr(self, "last_odom") and self.last_odom is not None:
            quaternion = Quaternion(
                self.last_odom.pose.pose.orientation.w,
                self.last_odom.pose.pose.orientation.x,
                self.last_odom.pose.pose.orientation.y,
                self.last_odom.pose.pose.orientation.z,
            )
            euler = quaternion.to_euler(degrees=False)
            angle = euler[2]
            density = self.compute_forward_obstacle_density(angle)
        else:
            density = 0
        density_penalty = 2.0 * density  # 系数可调
        # === END ===
        return base + obs_penalty + density_penalty


