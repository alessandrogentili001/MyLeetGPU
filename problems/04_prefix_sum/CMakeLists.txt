add_executable(prefix_sum
    main.cu
    kernel.cu
)

target_include_directories(prefix_sum PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
