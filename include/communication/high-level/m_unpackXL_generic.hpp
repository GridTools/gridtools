/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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
#include "wrap_argument.hpp"

template < typename value_type >
__global__ void m_unpackXLKernel_generic(value_type *__restrict__ d_data,
    value_type **d_msgbufTab_r,
    const wrap_argument d_msgsize_r,
    const gridtools::array< gridtools::halo_descriptor, 3 > halo /*_g*/,
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

    int isrc = oa + ob * la + oc * la * lb + d_msgsize_r[b_ind];

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
        // printf("XL %d %d %d -> %16.16e\n", idx, idy, idz, x);
        d_data[idst] = x;
    }
}

template < typename array_t >
void m_unpackXL_generic(array_t &fields, typename array_t::value_type::value_type **d_msgbufTab_r, int *d_msgsize_r) {

#ifdef CUDAMSG
    // just some timing stuff
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
#endif
    const int niter = fields.size();

    // run the compression a few times, just to get a bit
    // more statistics
    for (int i = 0; i < niter; i++) {

        // threads per block. Should be at least one warp in x, could be wider in y
        const int ntx = 1;
        const int nty = 32;
        const int ntz = 8;
        dim3 threads(ntx, nty, ntz);

        // form the grid to cover the entire plane. Use 1 block per z-layer
        int nx = fields[i].halos[0].r_length(-1);
        int ny = fields[i].halos[1].r_length(0);
        int nz = fields[i].halos[2].r_length(0);

        int nbx = (nx + ntx - 1) / ntx;
        int nby = (ny + nty - 1) / nty;
        int nbz = (nz + ntz - 1) / ntz;
        dim3 blocks(nbx, nby, nbz);

#ifdef CUDAMSG
        printf("UNPACK XL Launch grid (%d,%d,%d) with (%d,%d,%d) threads (full size: %d,%d,%d)\n",
            nbx,
            nby,
            nbz,
            ntx,
            nty,
            ntz,
            nx,
            ny,
            nz);
#endif

        if (nbx != 0 && nby != 0 && nbz != 0) {
            // the actual kernel launch
            // clang-format off
        m_unpackXLKernel_generic<<<blocks, threads, 0, XL_stream>>>
        (fields[i].ptr,
         reinterpret_cast<typename array_t::value_type::value_type**>(d_msgbufTab_r),
         wrap_argument(d_msgsize_r+27*i),
         *(reinterpret_cast<const gridtools::array<gridtools::halo_descriptor,3>*>(&fields[i])),
         ny, nz,
         (fields[i].halos[0].begin()-fields[i].halos[0].minus())
         + (fields[i].halos[1].begin())*fields[i].halos[0].total_length()
         + (fields[i].halos[2].begin())*fields[i].halos[0].total_length() *fields[i].halos[1].total_length(), 0);
// clang-format on

#ifdef CUDAMSG
            int err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("KLF in %s\n", __FILE__);
                exit(-1);
            }
#endif
        }
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

    // double nnumb =  niter * (double) (nx * ny * nz);
    // double nbyte =  nnumb * sizeof(double);

    // printf("XL Packed %g numbers in %g ms, BW = %g GB/s\n",
    //     nnumb, elapsedTime, (nbyte/(elapsedTime/1e3))/1e9);

    printf("XL Packed numbers in %g ms\n", elapsedTime);
#endif
}
