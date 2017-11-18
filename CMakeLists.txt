cmake_minimum_required(VERSION 2.8)

project(ba_demo_ceres)

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")


include_directories(${CERES_INCLUDE_DIRS} )
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ba_demo.cpp)
target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES} )

add_executable(ba_6d ba_demo_6d.cpp)
target_link_libraries(ba_6d ${CERES_LIBRARIES})

add_executable(curve_fit curve_fit.cc)
target_link_libraries(curve_fit ${CERES_LIBRARIES} ${OpenCV_LIBS})
