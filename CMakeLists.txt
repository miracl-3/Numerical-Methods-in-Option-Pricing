cmake_minimum_required(VERSION 3.16)
project(option_pricing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PRICING_METHODS binomial_european binomial_american monte_carlo_european
    monte_carlo_american variance_reduction lsm_american)
foreach(method IN LISTS PRICING_METHODS)
    add_executable(${method} src/${method}.cpp)
    target_include_directories(${method} PRIVATE include)
    if(MSVC)
        target_compile_options(${method} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${method} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endforeach()

include(CTest)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_custom_target(reproducibility_metadata ALL
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/build.py --record-only
            --compiler ${CMAKE_CXX_COMPILER} --build-dir $<TARGET_FILE_DIR:binomial_american>
        DEPENDS ${PRICING_METHODS}
        VERBATIM)
endif()
if(BUILD_TESTING)
    if(Python3_Interpreter_FOUND)
        add_test(NAME numerical_validation
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_pricing.py
                --build-dir $<TARGET_FILE_DIR:binomial_european>)
        add_test(NAME original_lsm_procedure
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_lsm_procedure.py)
        add_test(NAME research_file_management
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_experiment_files.py
                --build-dir $<TARGET_FILE_DIR:binomial_american>)
    endif()
endif()
