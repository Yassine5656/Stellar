cmake_minimum_required(VERSION 3.20)
project(Stellar VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_library(tensor STATIC libstellar/core/tensor.cc)

target_link_libraries(tensor PUBLIC OpenMP::OpenMP_CXX)

target_include_directories(tensor PUBLIC libstellar/core)

add_executable(stellar src/stellar.cc
        tests/core/tensor_tests.cc)

if (MSVC)
    target_compile_options(tensor PRIVATE /W4 /permissive-)
else ()
    target_compile_options(tensor PUBLIC -O3 -march=native -fno-finite-math-only -fno-trapping-math -fopenmp)
    target_compile_options(stellar PRIVATE -Wall -Wextra -O3)
endif ()

target_link_libraries(stellar PRIVATE tensor)


enable_testing()

add_executable(testsuite)

target_sources(testsuite
        PRIVATE tests/core/tensor_tests.cc
        PRIVATE libstellar/core/tensor.cc)

