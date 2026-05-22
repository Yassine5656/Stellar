# Copyright 2026 Yassine TLICH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)

add_library(libstellar SHARED)

target_sources(libstellar PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/misc.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/tensor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/init.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/io.cc)

target_include_directories(libstellar
            PUBLIC ${CMAKE_SOURCE_DIR}/include
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(OpenMP REQUIRED)
target_link_libraries(libstellar PRIVATE OpenMP::OpenMP_CXX)

if (MSVC)
    target_compile_options(libstellar PRIVATE /W4 /permissive-)
else ()
    target_compile_options(libstellar PUBLIC -O3 -march=native -fno-finite-math-only -fno-trapping-math -fopenmp)
endif()