cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(maskDetection)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(xtensor REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

message(STATUS "Pytorch status:")
message(STATUS "  libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "  version: ${OpenCV_VERSION}")
message(STATUS "  libraries: ${OpenCV_LIBS}")
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(maskDetection src/maskDetection.cpp src/anchors.cpp src/anchors.h src/nms.h src/nms.cpp)
target_link_libraries(maskDetection ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET maskDetection PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)