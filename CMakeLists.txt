cmake_minimum_required(VERSION 3.24)
project(Games101)

set(OpenCV_DIR "F:/WorkSpace/ThirdParty/opencv-4.7.0/x64/mingw/lib")
include_directories("F:/WorkSpace/ThirdParty")

# add_subdirectory(source/HW0)
# add_subdirectory(source/HW1)
#add_subdirectory(source/HW2)
add_subdirectory(source/HW3)
add_executable(test source/test.cpp)

