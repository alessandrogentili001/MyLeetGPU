add_executable(stencil_conv
    main.cu
    kernel.cu
)

target_include_directories(stencil_conv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
