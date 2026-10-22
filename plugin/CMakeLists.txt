# libsimpleshower_c8: a SimpleShower bundle as a CORSIKA 8 surrogate model.
#
#   cmake -S plugin -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-<version>
#   cmake --build build
#
# ONNXRUNTIME_ROOT is an ONNX Runtime release (include/ and lib/); or give
# ONNXRUNTIME_INCLUDE_DIR (where onnxruntime_cxx_api.h is) and ONNXRUNTIME_LIBRARY (the
# libonnxruntime.so file). Nothing of CORSIKA is needed but c8_surrogate.h, copied here.
cmake_minimum_required (VERSION 3.16)
project (simpleshower_c8 LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release)
endif ()

set (ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime release directory")
find_path (ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/include
  PATH_SUFFIXES onnxruntime onnxruntime/core/session)
find_library (ONNXRUNTIME_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib)
if (NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message (FATAL_ERROR "ONNX Runtime not found: set ONNXRUNTIME_ROOT, or "
                      "ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY")
endif ()

add_library (simpleshower_c8 MODULE simpleshower_c8.cpp)
target_include_directories (simpleshower_c8 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                                    ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries (simpleshower_c8 PRIVATE ${ONNXRUNTIME_LIBRARY})
get_filename_component (ONNXRUNTIME_LIBRARY_DIR ${ONNXRUNTIME_LIBRARY} DIRECTORY)
set_target_properties (simpleshower_c8 PROPERTIES
  PREFIX "lib"
  CXX_VISIBILITY_PRESET hidden   # only c8s_get_plugin is exported
  VISIBILITY_INLINES_HIDDEN ON
  BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR}"
  INSTALL_RPATH "$ORIGIN")
install (TARGETS simpleshower_c8 LIBRARY DESTINATION lib)
