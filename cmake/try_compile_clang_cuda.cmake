# Tries CUDA compilation using clang
#
# Usage of this function:
#
#  try_compile_clang_cuda(CUDA_ARCH)
#
#  Parameters:
#    - CUDA_ARCH: architecture used for testing compilation
#                 (doesn't have to match the architecture of an available GPU, no code is executed)
#
# This function defines:
#  GT_CLANG_CUDA_WORKS         CUDA test file was successfully compiled with clang
#
function(try_compile_clang_cuda CUDA_ARCH)
    set(CLANG_CUDA_TEST_SOURCE
"
__global__ void helloworld(int* in, int* out) {
    *out = *in;
}
int main(int argc, char* argv[]) {
    int* in;
    int* out;
    cudaMalloc((void**)&in, sizeof(int));
    cudaMalloc((void**)&out, sizeof(int));
    helloworld<<<1,1>>>(in, out);
    cudaFree(in);
    cudaFree(out);
}
")

    set(SRC_FILE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/TryClangCuda.cpp)
    file(WRITE "${SRC_FILE}" "${CLANG_CUDA_TEST_SOURCE}")

    set(CLANG_CUDA_FLAGS "-xcuda --cuda-path=${CUDAToolkit_BIN_DIR}/.. --cuda-gpu-arch=${CUDA_ARCH}")
    try_compile(clang_cuda_works ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY} ${SRC_FILE}
        COMPILE_DEFINITIONS "${CLANG_CUDA_FLAGS}"
        LINK_LIBRARIES CUDA::cudart
        OUTPUT_VARIABLE CLANG_CUDA_TRY_COMPILE_OUTPUT
        )
    if(NOT clang_cuda_works)
        file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
            "Testing CUDA compilation with clang gave the following output:\n${CLANG_CUDA_TRY_COMPILE_OUTPUT}\n\n")
    endif()
    set(GT_CLANG_CUDA_WORKS ${clang_cuda_works} PARENT_SCOPE)
endfunction()
