add_executable(matrix_transpose
    main.cu
    kernel.cu
)

target_include_directories(matrix_transpose PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
