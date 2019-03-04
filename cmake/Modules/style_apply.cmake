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

macro(SUBDIRLIST SUBDIR_LIST SUBDIR)
    message(STATUS "Starting at ${SUBDIR}")
    list(APPEND SUBDIR_LIST "${SUBDIR}")
    message(STATUS "SUBDIR_LISTxxx is "${SUBDIR_LIST})
    file(GLOB CHILDREN RELATIVE ${SUBDIR} ${SUBDIR}/*)
    foreach(CHILD ${CHILDREN})
        if(IS_DIRECTORY ${SUBDIR}/${CHILD})
            SUBDIRLIST(SUBDIR_LIST, "${SUBDIR}/${CHILD}")
        endif()
    endforeach()
endmacro()

function(STYLE_APPLY_DIR PATH)
    message(STATUS ${PATH})
    # execute_process(COMMAND ${CLANG_FORMAT} -i *.?pp
    #     WORKING_DIRECTORY ${PATH})
endfunction()

set(DIRECTORIES_OF_INTEREST
    src
    doc
    test
    python/pyngraph
)

set(CLANG_FORMAT_FILENAME clang-format-3.9)
find_program(CLANG_FORMAT ${CLANG_FORMAT_FILENAME} PATHS ENV PATH)

if (CLANG_FORMAT)
    foreach(DIRECTORY ${DIRECTORIES_OF_INTEREST})
        set(SUBDIRS "")
        subdirlist(SUBDIRS "${NGRAPH_SOURCE_DIR}/${DIRECTORY}")
        foreach(DIR ${SUBDIRS})
            style_apply_dir("${NGRAPH_SOURCE_DIR}/${DIR}")
        endforeach()
        # set(DIR "${NGRAPH_SOURCE_DIR}/${DIRECTORY}/*.?pp")
        # file(GLOB_RECURSE XPP_FILES ${DIR})
        # foreach(FILE ${XPP_FILES})
        #     style_apply_file(${FILE})
        # endforeach(FILE)
    endforeach(DIRECTORY)
else()
    message(STATUS "${CLANG_FORMAT_FILENAME} not found, style not available")
endif()
