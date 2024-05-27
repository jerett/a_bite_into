cmake_minimum_required(VERSION 3.0)
project(DeepLearning)

set(CMAKE_CXX_STANDARD 11)

# 查找OpenCV包
find_package(OpenCV REQUIRED)

# 查找Eigen包
find_package(Eigen3 REQUIRED)

# 包含OpenCV和Eigen头文件目录
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# 将src目录下的所有头文件和源文件添加到变量SRC_FILES中
file(GLOB_RECURSE SRC_FILES "src/*.h" "src/*.cc")

# 创建一个静态库，使用src目录下的文件
add_library(SrcLib STATIC ${SRC_FILES})
target_include_directories(SrcLib PUBLIC src)

# 设置源文件的组织结构
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC_FILES})

add_subdirectory(samples)
