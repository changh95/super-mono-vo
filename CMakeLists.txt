cmake_minimum_required(VERSION 2.8)
project( mono-vo )
find_package(OpenCV REQUIRED HINTS /home/hyunggi/mono-vo/opencv/share/OpenCV)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_subdirectory(superpointglue)

file(GLOB viso
    "src/*.h"
    "src/*.cpp"
)
add_executable( vo ${viso} )
target_link_libraries( vo ${OpenCV_LIBS} )
