# Find Openblas libraries
# Reference://github.com/BVLC/caffe/blob/master/cmake/Modules/FindOpenBLAS.cmake 
#
# If found, the following variable will be set.
#    OPENBLAS_FOUND     # ON
#    OPENBLAS_INC_DIR   # the include directory for cblas.
#    OPENBLAS_LIBS      # a list of libraries should be linked by paddle.
#                       # Each library should be full path to object file.

set(OPENBLAS_FOUND OFF)

set(OPENBLAS_ROOT $ENV{OPENBLAS_ROOT} CACHE PATH "Folder contains Openblas")
set(OPENBLAS_INCLUDE_SEARCH_PATHS
        ${OPENBLAS_ROOT}/include
        /usr/include
        /usr/include/openblas
        /opt/openblas/include
        /usr/local/include
        /usr/local/include/openblas
        /usr/local/opt/openblas/include
)

set(OPENBLAS_LIB_SEARCH_PATHS
        ${OPENBLAS_ROOT}/lib
        /lib
        /usr/lib
        /usr/lib64
        /usr/lib/blas/openblas
        /usr/lib/openblas
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /usr/local/opt/openblas/lib
        /opt/openblas/lib
)

find_path(OPENBLAS_INC_DIR NAMES cblas.h
  PATHS ${OPENBLAS_INCLUDE_SEARCH_PATHS})
find_library(OPENBLAS_LIB NAMES openblas
  PATHS ${OPENBLAS_LIB_SEARCH_PATHS})

if(OPENBLAS_INC_DIR AND OPENBLAS_LIB)
  set(OPENBLAS_FOUND ON)
  set(OPENBLAS_INC_DIR ${OPENBLAS_INC_DIR})
  set(OPENBLAS_LIBRARIES ${OPENBLAS_LIB})

  add_definitions(-DPADDLE_USE_OPENBLAS)

  message(STATUS "Found OpenBLAS (include: ${OPENBLAS_INC_DIR}, library: ${OPENBLAS_LIBRARIES})")
endif()
