export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export ROS_PORT_SIM=11311
export GAZEBO_RESOURCE_PATH=/root/RosRL/catkin_ws/src/multi_robot_scenario/launch
source ~/.bashrc
conda deactivate
cd ~/RosRL/catkin_ws

# 检查 ROS 环境变量是否已正确设置
if [ -z "$ROS_HOSTNAME" ] || [ -z "$ROS_MASTER_URI" ]; then
    echo "Error: ROS 环境变量未正确设置"
    exit 1
fi

# 检查 catkin_ws 是否存在
if [ ! -d "/root/RosRL/catkin_ws" ]; then
    echo "Error: catkin_ws 目录不存在"
    exit 1
fi

# 加载 ROS 环境
source /root/RosRL/catkin_ws/devel_isolated/setup.bash || {
    echo "Error: 无法加载 ROS 环境"
    exit 1
}

cd ..
cd TD3/

# 如果脚本执行成功，输出提示信息
echo "ROS 环境已成功加载，catkin_ws 目录已准备好。"
