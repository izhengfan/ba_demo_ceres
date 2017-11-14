cmake_minimum_required(VERSION 2.8)

project(ba_demo_ceres)

find_package(Ceres REQUIRED)

set(CMAKE_BUILD_TYPE "Release") 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")


include_directories(${CERES_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} ba_demo.cpp)

target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES} )
