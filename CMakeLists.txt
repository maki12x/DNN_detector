cmake_minimum_required(VERSION 2.8.3)
project(detector)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  mavros
  mavros_msgs
  roscpp
  rospy
  std_msgs
)

find_package (GeographicLib 1.34 REQUIRED)

find_package (OpenCV 3.4.0 REQUIRED)
message("OpenCV version: ${OpenCV_VERSION}")
include_directories(
include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   mavros_msgs
   std_msgs
 )


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES mapping
   CATKIN_DEPENDS mavros mavros_msgs roscpp rospy std_msgs
#  DEPENDS system_lib
)

add_executable(detector src/mavrosCommandMain.cpp src/mavrosCommand.cpp)
target_link_libraries(detector ${catkin_LIBRARIES} ${OpenCV_LIBS})


