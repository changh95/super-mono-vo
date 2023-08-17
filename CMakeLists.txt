cmake_minimum_required(VERSION 2.8)
project( super-mono-vo )

add_definitions(-w)

add_subdirectory(superpointglue)

find_package(OpenCV 4.2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/superpointglue/include
        ${PROJECT_SOURCE_DIR}/superpointglue/3rdparty/tensorrtbuffer/include
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
        )

file(GLOB viso
    "src/*.h"
    "src/*.cpp"
)
add_executable( vo ${viso} )
target_link_libraries( vo ${OpenCV_LIBS} superpointglue_lib )
