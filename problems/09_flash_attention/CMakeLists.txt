add_executable(flash_attention
    main.cu
    kernel.cu
)

target_include_directories(flash_attention PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
