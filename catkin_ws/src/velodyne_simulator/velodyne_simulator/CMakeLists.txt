cmake_minimum_required(VERSION 3.0.2)
project(velodyne_simulator)

# 在 CMakeLists.txt 文件中设置 Python 解释器路径
set(PythonInterp "/bin/python3" CACHE FILEPATH "Path to Python interpreter")
set(PYTHON_EXECUTABLE "/bin/python3" CACHE FILEPATH "Path to Python executable")

find_package(catkin REQUIRED)
catkin_metapackage()
