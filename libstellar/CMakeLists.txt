cmake_minimum_required(VERSION 3.20)
project(Stellar VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(libstellar SHARED)

target_sources(libstellar PRIVATE core/tensor.cc)

target_include_directories(libstellar
            PUBLIC ${CMAKE_SOURCE_DIR}/include
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(OpenMP REQUIRED)
target_link_libraries(libstellar PRIVATE OpenMP::OpenMP_CXX)

if (MSVC)
    target_compile_options(libstellar  PRIVATE /W4 /permissive-)
else ()
    target_compile_options(libstellar PUBLIC -O3 -march=native -fno-finite-math-only -fno-trapping-math -fopenmp)
endif()