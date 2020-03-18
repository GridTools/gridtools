# Tries CUDA compilation using clang
#
# Usage of this function:
#
#  try_compile_clang_cuda()
#
# This function defines:
#  GT_CLANG_CUDA_WORKS         CUDA test file was successfully compiled with clang
#
function(try_compile_clang_cuda)
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

    try_compile(clang_cuda_works ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY} ${SRC_FILE})
    set(GT_CLANG_CUDA_WORKS ${clang_cuda_works} PARENT_SCOPE)
endfunction()
