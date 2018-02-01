/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "interface/wrapper/gt_interface.h"
#ifdef C_INTERFACE_EXAMPLE_REPOSITORY
// TODO repository
#else
#include "example_wrapper_simple.h"
#endif

typedef struct wrappable {
} wrappable;

// C_INTERFACE_EXAMPLE_PADDING -> use an arbitrary (=1) padding
#include "c_interface_helper.h"

#ifdef CUDA_EXAMPLE
#include <cuda_runtime.h>
#endif

int main() {
    const int dim = 3;
    const int Nx = 4;
    const int Ny = 5;
    const int Nz = 6;

    int *dims = (int *)malloc(sizeof(int) * dim);
    int *strides = (int *)malloc(sizeof(int) * dim);
    int size;
    make_array_info(dims, strides, &size, Nx, Ny, Nz);

    DATA_TYPE *in = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * size);
    fill_array_unique(dims, strides, in);
    DATA_TYPE *out = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * size);
    fill_array(dims, strides, out, -1.);

#ifdef C_INTERFACE_EXAMPLE_REPOSITORY
    gt_handle *my_wrapper = alloc_simple_wrapper(dim, dims);
#else
    gt_handle *my_wrapper = alloc_simple_wrapper(dim, dims);
#endif
    printf("allocated wrapper\n");

#ifdef CUDA_EXAMPLE // TODO clean this (I don't want to use a .cu file because there is no cuda kernel stuff here, just
                    // calls to the cuda runtime API

    printf("pushing gpu ptrs\n");
    DATA_TYPE *dev_in;
    printf("%p\n", dev_in);
    cudaMalloc((void **)&dev_in, sizeof(DATA_TYPE) * size);
    printf("%p\n", dev_in);
    cudaMemcpy(dev_in, in, sizeof(DATA_TYPE) * size, cudaMemcpyHostToDevice);
    DATA_TYPE *dev_out;
    cudaMalloc((void **)&dev_out, sizeof(DATA_TYPE) * size);
    cudaMemcpy(dev_out, out, sizeof(DATA_TYPE) * size, cudaMemcpyHostToDevice);

    GT_PUSH(my_wrapper, "in", dev_in, dim, dims, strides, 0);
    GT_PUSH(my_wrapper, "out", dev_out, dim, dims, strides, 0);

    gt_run(my_wrapper);

    GT_PULL(my_wrapper, "out", dev_out, dim, dims, strides);

    cudaMemcpy(in, dev_in, sizeof(DATA_TYPE) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(out, dev_out, sizeof(DATA_TYPE) * size, cudaMemcpyDeviceToHost);

    cudaFree(dev_in);
    cudaFree(dev_out);
#else
    GT_PUSH(my_wrapper, "in", in, dim, dims, strides, 0);
    GT_PUSH(my_wrapper, "out", out, dim, dims, strides, 0);

    gt_run(my_wrapper);

    GT_PULL(my_wrapper, "out", out, dim, dims, strides);
#endif

    if (verify(dims, strides, in, out))
        printf("verified\n");
    else
        printf("failed\n");

    gt_release(my_wrapper); // TODO this doesn't currently delete the wrapper (only the handle around the raw ptr)

    free(in);
    free(out);

    free(dims);
    free(strides);
}
