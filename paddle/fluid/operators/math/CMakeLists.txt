function(math_library TARGET)
    # math_library is a function to create math library.
    # The interface is the same as cc_library.
    # But it handle split GPU/CPU code and link some common library.
    set(cc_srcs)
    set(cu_srcs)
    set(math_common_deps device_context tensor)
    set(multiValueArgs DEPS)
    cmake_parse_arguments(math_library "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN})

    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cc)
        list(APPEND cc_srcs ${TARGET}.cc)
    endif()
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cu)
        list(APPEND cu_srcs ${TARGET}.cu)
    endif()

    list(LENGTH cc_srcs cc_srcs_len)
    if (WITH_GPU)
        nv_library(${TARGET} SRCS ${cc_srcs} ${cu_srcs} DEPS ${math_library_DEPS} ${math_common_deps})
    else()
        cc_library(${TARGET} SRCS ${cc_srcs} DEPS ${math_library_DEPS} ${math_common_deps})
    endif()
endfunction()

math_library(selected_rows_functor DEPS selected_rows math_function)
cc_test(selected_rows_functor_test SRCS selected_rows_functor_test.cc DEPS selected_rows_functor)
if(WITH_GPU)
    nv_test(selected_rows_functor_gpu_test SRCS selected_rows_functor_test.cu DEPS selected_rows_functor)
endif()