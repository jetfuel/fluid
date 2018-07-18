add_subdirectory(math)
if (WITH_GPU)
    nv_library(adam_op SRCS adam_op.cc adam_op.cu DEPS op_registry selected_rows_functor)
else()
    cc_library(adam_op SRCS adam_op.cc DEPS op_registry selected_rows_functor)
endif ()


cc_test(adam_op_test SRCS adam_op_test.cc DEPS adam_op)
