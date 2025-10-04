cmake_minimum_required(VERSION 3.16)

project(RawDogML)

find_package(OpenMP REQUIRED)

file(GLOB_RECURSE MY_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable("${PROJECT_NAME}")

set_property(TARGET "${PROJECT_NAME}" PROPERTY CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)

#target_compile_definitions("${PROJECT_NAME}" PUBLIC RESOURCES_PATH="./resources/") # For running .exe
target_compile_definitions("${PROJECT_NAME}" PUBLIC RESOURCES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/resources/") # For development in IDE

target_sources("${PROJECT_NAME}" PRIVATE ${MY_SOURCES})

target_include_directories("${PROJECT_NAME}" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/")

target_link_libraries("${PROJECT_NAME}" PRIVATE OpenMP::OpenMP_CXX)