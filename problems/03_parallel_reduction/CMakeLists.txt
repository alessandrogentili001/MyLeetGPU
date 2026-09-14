add_executable(parallel_reduction
    main.cu
    kernel.cu
)

target_include_directories(parallel_reduction PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
