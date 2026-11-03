cmake_minimum_required(VERSION 3.15)

project(flimfit_labview C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(FLIMLIB_ROOT "${CMAKE_CURRENT_LIST_DIR}/flimlib_source" CACHE PATH "Path to FLIMLib source")
if(NOT EXISTS "${FLIMLIB_ROOT}/src/main/c/Ecf.h")
    message(FATAL_ERROR "Set FLIMLIB_ROOT to a FLIMLib source checkout")
endif()

add_library(flimfit_labview SHARED
    flimfit_labview.c
    "${FLIMLIB_ROOT}/src/main/c/EcfGlobal.c"
    "${FLIMLIB_ROOT}/src/main/c/EcfMultiple.c"
    "${FLIMLIB_ROOT}/src/main/c/EcfSPA.c"
    "${FLIMLIB_ROOT}/src/main/c/EcfSingle.c"
    "${FLIMLIB_ROOT}/src/main/c/EcfUtil.c"
    "${FLIMLIB_ROOT}/src/main/c/EcfWrapper.c"
    "${FLIMLIB_ROOT}/src/main/c/GCI_Lsqnonneg.c"
    "${FLIMLIB_ROOT}/src/main/c/GCI_Phasor.c"
)

target_include_directories(flimfit_labview PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${FLIMLIB_ROOT}/src/main/c"
)

if(MSVC)
    target_compile_definitions(flimfit_labview PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(flimfit_labview PRIVATE /O2 /W4)
    target_link_options(flimfit_labview PRIVATE /guard:cf)
else()
    target_compile_options(flimfit_labview PRIVATE -O3)
    target_link_libraries(flimfit_labview PRIVATE m)
endif()
