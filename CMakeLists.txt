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

file(GLOB mono_vo_src
    "src/*.h"
    "src/visodo.cpp"
)
add_executable( mono-vo
        ${mono_vo_src}
)
target_link_libraries( mono-vo ${OpenCV_LIBS} superpointglue_lib )

file(GLOB super_vo_src
        "src/*.h"
        "src/super_visodo.cpp"
)
add_executable( super-vo
        ${super_vo_src}
)
target_link_libraries( super-vo ${OpenCV_LIBS} superpointglue_lib )
