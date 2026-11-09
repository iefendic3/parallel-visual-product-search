cmake_minimum_required(VERSION 3.16)
project(parallel_visual_product_search VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(ann_core STATIC src/ivf_flat.cpp)
target_include_directories(ann_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(MSVC)
    target_compile_options(ann_core PRIVATE /W4)
else()
    target_compile_options(ann_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
