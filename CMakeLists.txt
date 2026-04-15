cmake_minimum_required(VERSION 3.20)
project(Stellar VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(libstellar)

add_executable(stellar)
target_sources(stellar PRIVATE src/stellar.cc)

target_compile_options(stellar PRIVATE -Wall -Wextra -O3)

target_link_libraries(stellar PRIVATE libstellar)