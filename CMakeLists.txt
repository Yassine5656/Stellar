cmake_minimum_required(VERSION 3.20)
project(Stellar VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_library(tensor STATIC src/core/tensor.cpp)

target_link_libraries(tensor PUBLIC OpenMP::OpenMP_CXX)

target_include_directories(tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/core)

add_executable(stellar src/stellar.cpp)

if (MSVC)
    target_compile_options(tensor PRIVATE /W4 /permissive-)
else ()
    target_compile_options(tensor PUBLIC -O3 -march=native -ffast-math -fopenmp)
    target_compile_options(stellar PRIVATE -Wall -Wextra -O3)
endif ()

target_link_libraries(stellar PRIVATE tensor)
