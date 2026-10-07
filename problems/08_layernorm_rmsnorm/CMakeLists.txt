add_executable(layernorm_rmsnorm
    main.cu
    kernel.cu
)

target_include_directories(layernorm_rmsnorm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
