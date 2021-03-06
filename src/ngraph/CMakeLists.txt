# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

set (SRC
    axis_set.cpp
    axis_vector.cpp
    autodiff/adjoints.cpp
    builder/autobroadcast.cpp
    builder/numpy_transpose.cpp
    builder/quantization/quantized_linear_convolution.cpp
    builder/quantization.cpp
    builder/reduce_ops.cpp
    coordinate.cpp
    coordinate_diff.cpp
    coordinate_transform.cpp
    descriptor/input.cpp
    descriptor/layout/dense_tensor_layout.cpp
    descriptor/layout/tensor_layout.cpp
    descriptor/output.cpp
    descriptor/tensor.cpp
    dimension.cpp
    file_util.cpp
    function.cpp
    log.cpp
    ngraph.cpp
    node.cpp
    op/abs.cpp
    op/acos.cpp
    op/add.cpp
    op/all.cpp
    op/allreduce.cpp
    op/and.cpp
    op/any.cpp
    op/argmin.cpp
    op/argmax.cpp
    op/asin.cpp
    op/atan.cpp
    op/avg_pool.cpp
    op/batch_norm.cpp
    op/broadcast.cpp
    op/ceiling.cpp
    op/concat.cpp
    op/constant.cpp
    op/convert.cpp
    op/convolution.cpp
    op/cos.cpp
    op/cosh.cpp
    op/dequantize.cpp
    op/divide.cpp
    op/dot.cpp
    op/embedding_lookup.cpp
    op/equal.cpp
    op/exp.cpp
    op/experimental/generate_mask.cpp
    op/experimental/quantized_avg_pool.cpp
    op/experimental/quantized_conv_bias.cpp
    op/experimental/quantized_conv_relu.cpp
    op/experimental/quantized_conv.cpp
    op/experimental/quantized_max_pool.cpp
    op/experimental/quantized_concat.cpp
    op/experimental/shape_of.cpp
    op/floor.cpp
    op/get_output_element.cpp
    op/greater.cpp
    op/greater_eq.cpp
    op/less.cpp
    op/less_eq.cpp
    op/log.cpp
    op/lrn.cpp
    op/max.cpp
    op/maximum.cpp
    op/max_pool.cpp
    op/min.cpp
    op/minimum.cpp
    op/multiply.cpp
    op/negative.cpp
    op/not.cpp
    op/not_equal.cpp
    op/one_hot.cpp
    op/op.cpp
    op/or.cpp
    op/pad.cpp
    op/parameter.cpp
    op/passthrough.cpp
    op/power.cpp
    op/product.cpp
    op/quantize.cpp
    op/relu.cpp
    op/replace_slice.cpp
    op/reshape.cpp
    op/result.cpp
    op/reverse.cpp
    op/reverse_sequence.cpp
    op/select.cpp
    op/sigmoid.cpp
    op/sign.cpp
    op/sin.cpp
    op/sinh.cpp
    op/slice.cpp
    op/softmax.cpp
    op/sqrt.cpp
    op/stop_gradient.cpp
    op/subtract.cpp
    op/sum.cpp
    op/tan.cpp
    op/tanh.cpp
    op/topk.cpp
    op/util/arithmetic_reduction.cpp
    op/util/binary_elementwise_arithmetic.cpp
    op/util/binary_elementwise_comparison.cpp
    op/util/binary_elementwise_logical.cpp
    op/util/index_reduction.cpp
    op/util/logical_reduction.cpp
    op/util/unary_elementwise_arithmetic.cpp
    partial_shape.cpp
    pass/algebraic_simplification.cpp
    pass/common_function_collection.cpp
    pass/constant_folding.cpp
    pass/cse.cpp
    pass/dump_sorted.cpp
    pass/get_output_element_elimination.cpp
    pass/graph_rewrite.cpp
    pass/like_replacement.cpp
    pass/liveness.cpp
    pass/manager.cpp
    pass/manager_state.cpp
    pass/memory_layout.cpp
    pass/memory_visualize.cpp
    pass/nop_elimination.cpp
    pass/pass.cpp
    pass/pass_config.cpp
    pass/prefix_reshape_elimination.cpp
    pass/propagate_cacheability.cpp
    pass/reshape_elimination.cpp
    pass/reshape_sinking.cpp
    pass/zero_dim_tensor_elimination.cpp
    pass/validate_graph.cpp
    pass/visualize_tree.cpp
    pass/core_fusion.cpp
    pass/serialize.cpp
    pass/zero_dim_tensor_elimination.cpp
    pattern/matcher.cpp
    runtime/aligned_buffer.cpp
    runtime/backend.cpp
    runtime/backend_manager.cpp
    runtime/executable.cpp
    runtime/host_tensor.cpp
    runtime/tensor.cpp
    serializer.cpp
    shape.cpp
    shape_util.cpp
    state/rng_state.cpp
    strides.cpp
    type/bfloat16.cpp
    type/element_type.cpp
    util.cpp
    validation_util.cpp
    graph_util.cpp
    placement.cpp
    cpio.cpp
    )

set(SRC ${SRC}
    runtime/hybrid/hybrid_backend.cpp
    runtime/hybrid/hybrid_executable.cpp
    runtime/hybrid/hybrid_util.cpp
    runtime/hybrid/op/function_call.cpp
    runtime/hybrid/pass/default_placement.cpp
    runtime/hybrid/pass/dump.cpp
    runtime/hybrid/pass/fix_get_output_element.cpp
    runtime/hybrid/pass/liveness.cpp
    runtime/hybrid/pass/memory_layout.cpp
    )


if(NGRAPH_DISTRIBUTED_ENABLE)
    list(APPEND SRC distributed.cpp)
endif()

configure_file(version.in.hpp version.hpp)

add_library(ngraph SHARED ${SRC})

if(NGRAPH_DISTRIBUTED_ENABLE)
    target_sources(ngraph PRIVATE distributed.cpp)
    if(NGRAPH_DISTRIBUTED_MLSL_ENABLE)
        target_include_directories(ngraph SYSTEM PRIVATE libmlsl)
        target_link_libraries(ngraph PRIVATE libmlsl)
    elseif(NGRAPH_DISTRIBUTED_OMPI_ENABLE)
        find_package(MPI REQUIRED)
        target_include_directories(ngraph SYSTEM PRIVATE ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
        target_link_libraries(ngraph PRIVATE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
    else()
        message(FATAL_ERROR "Distributed Library not supported/mentioned")
    endif()
endif()

add_subdirectory(frontend)

find_package(Graphviz QUIET)
if (GRAPHVIZ_FOUND)
    set_property(SOURCE pass/visualize_tree.cpp APPEND PROPERTY COMPILE_DEFINITIONS GRAPHVIZ_FOUND)
endif()

if(NGRAPH_ADDRESS_SANITIZER)
    message(STATUS "Enable Address Sanitizer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
endif()

add_subdirectory(codegen)
add_subdirectory(runtime)

target_compile_definitions(ngraph
    PRIVATE
        SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        SHARED_LIB_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}"
        NGRAPH_DLL_EXPORTS
)
if(NGRAPH_LIB_VERSIONING_ENABLE)
    set_target_properties(ngraph PROPERTIES
        VERSION ${NGRAPH_VERSION}
        SOVERSION ${NGRAPH_API_VERSION})
endif()
target_link_libraries(ngraph PRIVATE libjson)
target_compile_definitions(ngraph PUBLIC NGRAPH_VERSION="${NGRAPH_VERSION}")

if (LINUX)
    set_property(TARGET ngraph APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--rpath,$ORIGIN")

    # nGraph links against one or more libraries (ex. LLVM) but we don't want to
    # export these symbols as part of the DSO. This is a GNU ld (and derivatives) specific
    # option so making this portable is still an open issue. As a note for the future,
    # this is not an issue on Windows and LLVM's lld does support --exclude-libs.
    set_property(TARGET ngraph APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--exclude-libs,ALL")

    # GCC invokes the linker with --as-needed by default which doesn't work for us
    # because generated code needs to find symbols in these DSOs at runtime.
    # The fix below is temporary and will be removed once we find a better way
    # to do this because certain dependencies like the OpenMP runtime libraries
    # _do_ need to be linked with --as-needed with a higher priority for the
    # Intel OpenMP runtime so we don't mix libgomp and libiomp5
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set_property(TARGET ngraph APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--no-as-needed")
    endif()
elseif(APPLE)
    set_property(TARGET ngraph APPEND_STRING PROPERTY LINK_FLAGS " -Wl,-rpath,@loader_path")
endif()

# Defines macro in C++ to load backend plugin
target_include_directories(ngraph PUBLIC "${NGRAPH_INCLUDE_PATH}")
if (NOT WIN32)
    target_link_libraries(ngraph PUBLIC dl pthread)
endif()

if (NGRAPH_ONNX_IMPORT_ENABLE)
    target_sources(ngraph PRIVATE $<TARGET_OBJECTS:onnx_import_interface>)
    target_link_libraries(ngraph PRIVATE onnx_import)
endif()

# Build subdirectories for all build types on Windows
if(WIN32)
    foreach(BUILD_TYPE Release Debug RelWithDebInfo MinSizeRel)
        if(NOT EXISTS ${NGRAPH_BUILD_DIR}/${BUILD_TYPE})
            file(MAKE_DIRECTORY ${NGRAPH_BUILD_DIR}/${BUILD_TYPE})
        endif()
    endforeach()
endif()

#-----------------------------------------------------------------------------------------------
# Installation logic...
#-----------------------------------------------------------------------------------------------

# nGraph
install(FILES ${CMAKE_BINARY_DIR}/VERSION DESTINATION ${CMAKE_INSTALL_PREFIX})
install(TARGETS ngraph DESTINATION ${NGRAPH_INSTALL_LIB})  # libngraph.so
install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION "${NGRAPH_INSTALL_INCLUDE}/ngraph"
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
)
install(FILES ${CMAKE_BINARY_DIR}/src/ngraph/version.hpp
    DESTINATION "${NGRAPH_INSTALL_INCLUDE}/ngraph")

set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A most excellent graph library")
set(CPACK_PACKAGE_NAME "nGraph")
set(CPACK_PACKAGE_CONTACT "Robert Kimball")
# set(CPACK_PACKAGE_VERSION
set(CPACK_PACKAGE_VENDOR "Intel Nervana")
# set(CPACK_PACKAGE_DESCRIPTION_FILE "")

set(CPACK_PACKAGE_VERSION_MAJOR ${NGRAPH_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${NGRAPH_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${NGRAPH_VERSION_PATCH})
include(CPack)
