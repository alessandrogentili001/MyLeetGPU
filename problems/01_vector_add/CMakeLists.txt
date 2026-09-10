add_executable(vector_add
    main.cu
    kernel.cu
)

target_include_directories(vector_add PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
