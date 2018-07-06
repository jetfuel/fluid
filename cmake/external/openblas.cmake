# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

IF(USE_EIGEN_FOR_BLAS)
    return()
ENDIF(USE_EIGEN_FOR_BLAS)

INCLUDE(FindOpenBLAS)

IF(NOT ${OPENBLAS_FOUND})
    INCLUDE(ExternalProject)

    SET(OPENBLAS_SOURCES_DIR ${BAZEL_THIRD_PARTY_DIR}/openblas)
    SET(OPENBLAS_INSTALL_DIR ${OPENBLAS_SOURCES_DIR}/install/openblas)
    SET(OPENBLAS_INC_DIR "${OPENBLAS_INSTALL_DIR}/include" CACHE PATH "openblas include directory." FORCE)

    SET(OPENBLAS_LIBRARIES
        "${OPENBLAS_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX}"
        CACHE FILEPATH "openblas library." FORCE)

    ADD_DEFINITIONS(-DPADDLE_USE_OPENBLAS)

    SET(OPENBLAS_CC "${CMAKE_C_COMPILER} -Wno-unused-but-set-variable -Wno-unused-variable")
    SET(OPENBLAS_COMMIT "v0.2.20")

    IF(CMAKE_CROSSCOMPILING)
        SET(OPTIONAL_ARGS HOSTCC=${HOST_C_COMPILER})
        GET_FILENAME_COMPONENT(CROSS_SUFFIX ${CMAKE_C_COMPILER} DIRECTORY)
        SET(CROSS_SUFFIX ${CROSS_SUFFIX}/)
        IF(ANDROID)
            IF(ANDROID_ABI MATCHES "^armeabi(-v7a)?$")
                # use softfp
                SET(OPTIONAL_ARGS ${OPTIONAL_ARGS} TARGET=ARMV7 ARM_SOFTFP_ABI=1 USE_THREAD=0)
            ELSEIF(ANDROID_ABI STREQUAL "arm64-v8a")
                SET(OPTIONAL_ARGS ${OPTIONAL_ARGS} TARGET=ARMV8 BINARY=64 USE_THREAD=0)
            ENDIF()
        ELSEIF(IOS)
            IF(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
                SET(OPENBLAS_CC "${OPENBLAS_CC} ${CMAKE_C_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")
                SET(OPENBLAS_CC "${OPENBLAS_CC} -arch arm64")
                SET(OPTIONAL_ARGS ${OPTIONAL_ARGS} TARGET=ARMV8 BINARY=64 USE_THREAD=0 CROSS_SUFFIX=${CROSS_SUFFIX})
            ELSE()
                MESSAGE(FATAL_ERROR "OpenBLAS only support arm64 architectures on iOS. "
                       "You can set IOS_USE_VECLIB_FOR_BLAS=ON or USE_EIGEN_FOR_BLAS=ON to use other blas library instead.")
            ENDIF()
        ELSEIF(RPI)
            # use hardfp
            SET(OPTIONAL_ARGS ${OPTIONAL_ARGS} TARGET=ARMV7 USE_THREAD=0)
        ENDIF()
    ELSE()
        IF(APPLE)
            SET(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
            SET(OPENBLAS_CC "${CMAKE_C_COMPILER} -isysroot ${CMAKE_OSX_SYSROOT}")
        ENDIF()
        SET(OPTIONAL_ARGS "")
        IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^x86(_64)?$")
            SET(OPTIONAL_ARGS DYNAMIC_ARCH=1 NUM_THREADS=64)
        ENDIF()
    ENDIF()

    SET(COMMON_ARGS CC=${OPENBLAS_CC} NO_SHARED=1 NO_LAPACK=1 libs)

    ExternalProject_Add(
        extern_openblas
        ${EXTERNAL_PROJECT_LOG_ARGS}
        GIT_REPOSITORY      https://github.com/xianyi/OpenBLAS.git
        GIT_TAG             ${OPENBLAS_COMMIT}
        PREFIX              ${OPENBLAS_SOURCES_DIR}
        INSTALL_DIR         ${OPENBLAS_INSTALL_DIR}
        BUILD_IN_SOURCE     1
        BUILD_COMMAND       ${CMAKE_MAKE_PROGRAM} ${COMMON_ARGS} ${OPTIONAL_ARGS}
        INSTALL_COMMAND     ${CMAKE_MAKE_PROGRAM} install NO_SHARED=1 NO_LAPACK=1 PREFIX=<INSTALL_DIR> 
                            && rm -r ${OPENBLAS_INSTALL_DIR}/lib/cmake ${OPENBLAS_INSTALL_DIR}/lib/pkgconfig
        UPDATE_COMMAND      ""
        CONFIGURE_COMMAND   ""
    )
    ADD_LIBRARY(cblas STATIC IMPORTED GLOBAL)
    SET_PROPERTY(TARGET cblas PROPERTY IMPORTED_LOCATION ${OPENBLAS_LIBRARIES})
    ADD_DEPENDENCIES(cblas extern_openblas)
    LIST(APPEND external_project_dependencies cblas)
    INCLUDE_DIRECTORIES(${OPENBLAS_INC_DIR})
ENDIF(NOT ${OPENBLAS_FOUND})

