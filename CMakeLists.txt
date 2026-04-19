cmake_minimum_required(VERSION 3.20)
project(Stellar VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(libstellar)

add_executable(stellar)
target_sources(stellar PRIVATE src/stellar.cc)

target_compile_options(stellar PRIVATE -Wall -Wextra -O3)

target_link_libraries(stellar PRIVATE libstellar)

target_include_directories(stellar
                PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libstellar/include)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
    add_custom_target(copy_compile_commands ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_BINARY_DIR}/compile_commands.json"
                "${CMAKE_SOURCE_DIR}/compile_commands.json"
            COMMENT "Copying 'compile_commands' to help LSP"
            VERBATIM
            )
endif()