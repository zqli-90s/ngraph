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

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.0.0")
        # gtest has issues with this with v1.8.x
        # gtest issue is supposed to be addressed after v1.8.x
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-zero-as-null-pointer-constant")
    endif()
endif()

set(SRC
    algebraic_simplification.cpp
    all_close_f.cpp
    assertion.cpp
    build_graph.cpp
    builder_autobroadcast.cpp
    constant_folding.cpp
    control_dependencies.cpp
    coordinate.cpp
    copy.cpp
    core.cpp
    cpio.cpp
    cse.cpp
    element_type.cpp
    file_util.cpp
    includes.cpp
    input_output_assign.cpp
    main.cpp
    misc.cpp
    nop_elimination.cpp
    op.cpp
    partial_shape.cpp
    pass_liveness.cpp
    pass_manager.cpp
    pass_memory_layout.cpp
    pattern.cpp
    reshape_elimination.cpp
    reshape_sinking.cpp
    serialize.cpp
    shape.cpp
    tensor.cpp
    type_prop.cpp
    util.cpp
    uuid.cpp
    zero_dim_tensor_elimination.cpp
)

if(NOT WIN32 AND NGRAPH_TOOLS_ENABLE)
    list(APPEND SRC tools.cpp)
endif()

if(NGRAPH_DISTRIBUTED_ENABLE)
    list(APPEND SRC distributed_setup.cpp)
endif()

set_source_files_properties(includes.cpp PROPERTIES COMPILE_DEFINITIONS
    NGRAPH_INCLUDES="${PROJECT_SOURCE_DIR}/src/ngraph")

if (NGRAPH_INTERPRETER_ENABLE)
    list(APPEND SRC
        backend_debug_api.cpp
        builder.cpp
        backend_api.cpp
        hybrid_backend.cpp)
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} INTERPRETER)
endif()

if (NGRAPH_CPU_ENABLE)
    list(APPEND SRC core_fusion.cpp builder_quantization.cpp)
    list(APPEND SRC backend_performance.cpp cpu_fusion.cpp cpu_test.cpp cpu_debugger.cpp)
    if (NOT NGRAPH_DEX_ONLY)
        list(APPEND SRC cpu_codegen.cpp)
    endif()
    if (NGRAPH_HALIDE)
        list(APPEND SRC halide.cpp)
    endif()
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} CPU)
endif()

if(NGRAPH_GPU_ENABLE)
    list(APPEND SRC gpu_test.cpp gpu_fusion.cpp)
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} GPU)
endif()

if (NGRAPH_INTELGPU_ENABLE)
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} INTELGPU)
endif()

if (NGRAPH_GPUH_ENABLE)
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} GPUH)
endif()

if (NGRAPH_PLAIDML_ENABLE)
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} PlaidML)
endif()

if (NGRAPH_GENERIC_CPU_ENABLE)
    set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} GCPU)
endif()

add_subdirectory(models)
add_subdirectory(files)
add_subdirectory(util)

# backend specific test files must meet the following requirements:
# 1) The must be named <name>.in.cpp
# 2) They must be in the test directory
# 3) add the line `static string s_manifest = "${MANIFEST}";` to your cpp file
# All such files are configured via cmake which replaces all instances of cmake variables
# such as ${BACKEND_NAME} with their values, such as CPU, GPU, or INTERPRETER.
set(MULTI_TEST_SRC
    autodiff.in.cpp
    backend_all.in.cpp
    backend_any.in.cpp
    backend_binary_elementwise.in.cpp
    backend_broadcast.in.cpp
    backend_comparison.in.cpp
    backend_dot.in.cpp
    backend_embedding_lookup.in.cpp
    backend_one_hot.in.cpp
    backend_pool.in.cpp
    backend_reshape.in.cpp
    backend_sum.in.cpp
    backend_topk.in.cpp
    backend_arg_reduce.in.cpp
    backend_test.in.cpp
    backend_unary_elementwise.in.cpp
    convolution_test.in.cpp
)

if(NGRAPH_DISTRIBUTED_ENABLE)
    list(APPEND MULTI_TEST_SRC distributed.in.cpp)
endif()

if (NGRAPH_CPU_ENABLE)
    list(APPEND MULTI_TEST_SRC backend_graph_comparison.in.cpp)
endif()

if (NGRAPH_ONNX_IMPORT_ENABLE)
    list(APPEND MULTI_TEST_SRC onnx_import.in.cpp)
    if (NGRAPH_ONNXIFI_ENABLE)
        list(APPEND SRC onnxifi.cpp onnxifi_span.cpp)
    endif()
endif()

foreach(BACKEND_NAME ${ACTIVE_BACKEND_LIST})
    # Some---but not all---autodiff tests go through multiple iterations with
    # different random seeds. On the CPU backend this is currently very slow
    # because the autodiff tests recompile with each iteration. That behavior
    # can be changed, but it's a bit involved, so for the time being we just
    # reduce the number of test iterations on non-INTERPRETER backends.
    if(${BACKEND_NAME} MATCHES ^INTERPRETER$)
        set(TEST_LOOPS 100)
    else()
        set(TEST_LOOPS 2)
    endif()

    string(TOLOWER ${BACKEND_NAME} BACKEND_DIR)
    set(MANIFEST ${PROJECT_SOURCE_DIR}/src/ngraph/runtime/${BACKEND_DIR}/unit_test.manifest)

    foreach(TEST_SRC ${MULTI_TEST_SRC})
        string(REPLACE ".in." "_${BACKEND_NAME}." TARGET_NAME ${TEST_SRC})
        configure_file(${TEST_SRC} ${TARGET_NAME})
        set(SRC ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME} ${SRC})
    endforeach()

    message(STATUS "Adding unit test for backend ${BACKEND_NAME}")
endforeach()

add_executable(unit-test ${SRC})

target_include_directories(unit-test PRIVATE ".")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCURDIR=\\\"${CMAKE_CURRENT_SOURCE_DIR}\\\"")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DJSON_INCLUDES=\\\"${JSON_INCLUDE_DIR}\\\"")

if(NGRAPH_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
endif()

if(NGRAPH_DISTRIBUTED_ENABLE)
    if(NGRAPH_DISTRIBUTED_MLSL_ENABLE)
        target_link_libraries(unit-test PRIVATE libmlsl)
    elseif(NGRAPH_DISTRIBUTED_OMPI_ENABLE)
        find_package(MPI REQUIRED)
        target_include_directories(unit-test
        SYSTEM PRIVATE ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
        target_link_libraries(unit-test PRIVATE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
    else()
            message(FATAL_ERROR "Distributed Library not supported/mentioned")
    endif()
endif()

target_link_libraries(unit-test PRIVATE ngraph_test_util)
target_link_libraries(unit-test PRIVATE ngraph libgtest libjson)
if(NOT WIN32)
    target_link_libraries(unit-test PRIVATE pthread)
endif()
target_link_libraries(unit-test PRIVATE ${CMAKE_DL_LIBS})

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(unit-test PRIVATE -Wno-undef -Wno-reserved-id-macro)
endif()

if (NGRAPH_CPU_ENABLE)
    # The INTERPRETER backend is required for convolution, and backwards unit tests
    target_link_libraries(unit-test PRIVATE cpu_backend interpreter_backend)
    target_link_libraries(unit-test PRIVATE libmkldnn)
endif()

if (NGRAPH_TOOLS_ENABLE)
    get_property(NBENCH_PATH TARGET nbench PROPERTY BINARY_DIR)
    set(NBENCH "${NBENCH_PATH}/nbench")
    target_compile_definitions(unit-test PRIVATE NBENCH_PATH="${NBENCH}")
    add_dependencies(unit-test nbench)
endif()

if (NGRAPH_PLAIDML_ENABLE)
    target_link_libraries(unit-test PRIVATE plaidml_backend)
endif()

if (NGRAPH_TBB_ENABLE)
    target_compile_definitions(unit-test PRIVATE NGRAPH_TBB_ENABLE)
endif()

if (NGRAPH_HALIDE)
    target_compile_definitions(unit-test PRIVATE "NGRAPH_HALIDE")
endif()

if (NGRAPH_INTERPRETER_ENABLE)
    target_compile_definitions(unit-test PRIVATE NGRAPH_INTERPRETER_ENABLE)
    target_link_libraries(unit-test PRIVATE interpreter_backend)
endif()

if (NGRAPH_GPU_ENABLE)
    target_link_libraries(unit-test PRIVATE gpu_backend)
endif()

if (NGRAPH_NOP_ENABLE)
    target_link_libraries(unit-test PRIVATE nop_backend)
endif()

if (NGRAPH_GPUH_ENABLE)
    target_link_libraries(unit-test PRIVATE gpuh_backend)
endif()

if (NGRAPH_ONNXIFI_ENABLE)
    target_include_directories(unit-test SYSTEM PUBLIC ${ONNX_INCLUDE_DIR})
    target_link_libraries(unit-test PRIVATE onnxifi-ngraph)
endif()

# If all the runtime libraries are installed into one location, that will make life easier.
if (MSVS)
    add_custom_target(unit-test-check
        COMMAND set "PATH=${EXTERNAL_PROJECTS_ROOT}/src/ngraph/Release;${EXTERNAL_PROJECTS_ROOT}/mkldnn/lib/;${EXTERNAL_PROJECTS_ROOT}/mkl/src/ext_mkl/lib/;${EXTERNAL_PROJECTS_ROOT}/ext_tbb-prefix/src/ext_tbb/tbb2019_20181203oss/bin/intel64/vc14;%PATH%"
        COMMAND ${PROJECT_BINARY_DIR}/test/unit-test \${ARGS}
        DEPENDS unit-test
    )
else()
    add_custom_target(unit-test-check
        COMMAND ${PROJECT_BINARY_DIR}/test/unit-test \${ARGS}
        DEPENDS unit-test
    )
endif()

add_custom_target(check
    DEPENDS
    style-check
    unit-test-check
)
