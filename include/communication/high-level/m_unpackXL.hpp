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
template < typename value_type >
__global__ void m_unpackXLKernel(value_type *__restrict__ d_data,
    value_type **d_msgbufTab_r,
    const int *d_msgsize_r,
    const gridtools::halo_descriptor *halo /*_g*/,
    int const ny,
    int const nz,
    const int traslation_const,
    const int field_index) {

    // per block shared buffer for storing destination buffers
    __shared__ value_type *msgbuf[27];
    //__shared__ gridtools::halo_descriptor halo[3];

    int idx = blockIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    // load msg buffer table into shmem. Only the first 9 threads
    // need to do this
    if (threadIdx.x == 0 && threadIdx.y < 27 && threadIdx.z == 0) {
        msgbuf[threadIdx.y] = d_msgbufTab_r[threadIdx.y];
    }

    int ba = 0;
    int la = halo[0].minus();

    int bb = 1;
    int lb = halo[1].end() - halo[1].begin() + 1;

    int bc = 1;

    int b_ind = ba + 3 * bb + 9 * bc;

    int oa = idx;
    int ob = idy;
    int oc = idz;

    int isrc = oa + ob * la + oc * la * lb + field_index * d_msgsize_r[b_ind];

    // at this point we need to be sure that threads 0 - 8 have loaded the
    // message buffer table.
    __syncthreads();

    value_type x;
    // store the data in the correct message buffer
    if ((idy < ny) && (idz < nz)) {
        x = msgbuf[b_ind][isrc];
    }

    // load the data from the contiguous source buffer
    int tli = halo[0].total_length();
    int tlj = halo[1].total_length();
    int idst = idx + idy * tli + idz * tli * tlj + traslation_const;

    if ((idy < ny) && (idz < nz)) {
        d_data[idst] = x;
    }
}

template < typename array_t, typename value_type >
void m_unpackXL(array_t &d_data_array,
    value_type **d_msgbufTab_r,
    int d_msgsize_r[27],
    const gridtools::halo_descriptor halo[3],
    const gridtools::halo_descriptor halo_d[3]) {
    // threads per block. Should be at least one warp in x, could be wider in y
    const int ntx = 1;
    const int nty = 32;
    const int ntz = 8;
    dim3 threads(ntx, nty, ntz);

    // form the grid to cover the entire plane. Use 1 block per z-layer
    int nx = halo[0].r_length(-1);
    int ny = halo[1].r_length(0);
    int nz = halo[2].r_length(0);

    int nbx = (nx + ntx - 1) / ntx;
    int nby = (ny + nty - 1) / nty;
    int nbz = (nz + ntz - 1) / ntz;
    dim3 blocks(nbx, nby, nbz);

    if (nbx == 0 || nby == 0 || nbz == 0)
        return;

#ifdef CUDAMSG
    printf("Launch grid (%d,%d,%d) with (%d,%d,%d) threads (full size: %d,%d,%d)\n",
        nbx,
        nby,
        nbz,
        ntx,
        nty,
        ntz,
        nx,
        ny,
        nz);

    // just some timing stuff
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
#endif
    const int niter = d_data_array.size();

    // run the compression a few times, just to get a bit
    // more statistics
    for (int i = 0; i < niter; i++) {

        // the actual kernel launch
        // clang-format off
      m_unpackXLKernel<<<blocks, threads, 0, XL_stream>>>(d_data_array[i], d_msgbufTab_r, d_msgsize_r, halo_d, ny, nz,
                                                          (halo[0].begin()-halo[0].minus())
                                                          + (halo[1].begin())*halo[0].total_length()
                                                          + (halo[2].begin())*halo[0].total_length() *halo[1].total_length(), i);
// clang-format on
#ifdef CUDAMSG
        int err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failure\n");
            exit(-1);
        }
#endif
    }

#ifdef CUDAMSG
    // more timing stuff and conversion into reasonable units
    // for display
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double nnumb = niter * (double)(nx * ny * nz);
    double nbyte = nnumb * sizeof(double);

    printf("XL Packed %g numbers in %g ms, BW = %g GB/s\n", nnumb, elapsedTime, (nbyte / (elapsedTime / 1e3)) / 1e9);
#endif
}

template < typename Blocks,
    typename Threads,
    typename Bytes,
    typename Pointer,
    typename MsgbufTab,
    typename Msgsize,
    typename Halo >
int call_kernel_XL_u(Blocks blocks,
    Threads threads,
    Bytes b,
    Pointer d_data,
    MsgbufTab d_msgbufTab,
    Msgsize d_msgsize,
    Halo halo_d,
    int nx,
    int ny,
    int tranlation_const,
    int i) {
    m_unpackXLKernel<<< blocks, threads, b, XL_stream >>>(
        d_data, d_msgbufTab, d_msgsize, halo_d, nx, ny, tranlation_const, i);

#ifdef CUDAMSG
    int err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failure\n");
        exit(-1);
    }
#endif

    return 0;
}

template < typename value_type, typename datas, unsigned int... Ids >
void m_unpackXL_variadic(value_type **d_msgbufTab_r,
    int d_msgsize_r[27],
    const gridtools::halo_descriptor halo[3],
    const gridtools::halo_descriptor halo_d[3],
    const datas &d_datas,
    gridtools::gt_integer_sequence< unsigned int, Ids... >) {
    // threads per block. Should be at least one warp in x, could be wider in y
    const int ntx = 1;
    const int nty = 32;
    const int ntz = 8;
    dim3 threads(ntx, nty, ntz);

    // form the grid to cover the entire plane. Use 1 block per z-layer
    int nx = halo[0].r_length(-1);
    int ny = halo[1].r_length(0);
    int nz = halo[2].r_length(0);

    int nbx = (nx + ntx - 1) / ntx;
    int nby = (ny + nty - 1) / nty;
    int nbz = (nz + ntz - 1) / ntz;
    dim3 blocks(nbx, nby, nbz);

    if (nbx == 0 || nby == 0 || nbz == 0)
        return;

#ifdef CUDAMSG
    printf("Launch grid (%d,%d,%d) with (%d,%d,%d) threads (full size: %d,%d,%d)\n",
        nbx,
        nby,
        nbz,
        ntx,
        nty,
        ntz,
        nx,
        ny,
        nz);

    // just some timing stuff
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
#endif

    const int niter = std::tuple_size< datas >::value;
    int nothing[niter] = {call_kernel_XL_u(blocks,
        threads,
        0,
        static_cast< value_type * >(std::get< Ids >(d_datas)),
        d_msgbufTab_r,
        d_msgsize_r,
        halo_d,
        ny,
        nz,
        (halo[0].begin() - halo[0].minus()) + (halo[1].begin()) * halo[0].total_length() +
            (halo[2].begin()) * halo[0].total_length() * halo[1].total_length(),
        Ids)...};

#ifdef CUDAMSG
    // more timing stuff and conversion into reasonable units
    // for display
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double nnumb = niter * (double)(nx * ny * nz);
    double nbyte = nnumb * sizeof(double);

    printf("XL Packed %g numbers in %g ms, BW = %g GB/s\n", nnumb, elapsedTime, (nbyte / (elapsedTime / 1e3)) / 1e9);
#endif
}
