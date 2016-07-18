/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
__global__ void m_packZLKernel_generic(const value_type *__restrict__ d_data,
    value_type **__restrict__ d_msgbufTab,
    const wrap_argument d_msgsize,
    const gridtools::array< gridtools::halo_descriptor, 3 > halo /*_g*/,
    int const nx,
    int const ny,
    int const field_index) {

    // per block shared buffer for storing destination buffers
    __shared__ value_type *msgbuf[27];
    //__shared__ gridtools::halo_descriptor halo[3];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z;

    // load msg buffer table into shmem. Only the first 9 threads
    // need to do this
    if (threadIdx.x < 27 && threadIdx.y == 0) {
        msgbuf[threadIdx.x] = d_msgbufTab[threadIdx.x];
    }

    // an expression used later quite a bit
    int aa = halo[0].plus() + halo[0].end() - halo[0].begin();
    int ab = halo[1].plus() + halo[1].end() - halo[1].begin();

    // load the data from the contiguous source buffer
    value_type x;
    int pas = 0, pbs = 0, pcs = 0;

    pas = halo[0].plus();
    if (idx < halo[0].plus())
        pas = 0;

    pbs = halo[1].plus();
    if (idy < halo[1].plus())
        pbs = 0;

    pcs = 0;

    int mas = 0;
    if (idx > aa)
        mas = halo[0].minus();

    //     int mas = (idx<=aa)?0:halo[0].minus();

    int mbs = 0;
    if (idy > ab)
        mbs = halo[1].minus();

    int mcs = 0;

    int ia = idx + halo[0].begin() - pas - mas;
    int ib = idy + halo[1].begin() - pbs - mbs;
    int ic = idz + halo[2].begin() - pcs - mcs;
    int isrc = ia + ib * halo[0].total_length() + ic * halo[0].total_length() * halo[1].total_length();

    if ((idx < nx) && (idy < ny)) {
        x = d_data[isrc];
        // printf("ZL %e\n", x);
    }

    int ba = 1;
    int aas = 0;
    int la = halo[0].end() - halo[0].begin() + 1;
    if (idx < halo[0].plus()) {
        ba = 0;
        la = halo[0].plus();
    }
    if (idx > aa) {
        ba = 2;
        la = halo[0].minus();
        aas = halo[0].end() - halo[0].begin() + 1;
    }

    int bb = 1;
    int abs = 0;
    int lb = halo[1].end() - halo[1].begin() + 1;
    if (idy < halo[1].plus()) {
        bb = 0;
        lb = halo[1].plus();
    }
    if (idy > ab) {
        bb = 2;
        lb = halo[1].minus();
        abs = halo[1].end() - halo[1].begin() + 1;
    }

    int bc = 0;
    int lc = halo[2].plus();

    int oa = idx - pas - aas;
    int ob = idy - pbs - abs;
    int oc = idz;

    int buff_index = ba + 3 * bb + 9 * bc;
    int idst = oa + ob * la + oc * la * lb + d_msgsize[buff_index];

    // at this point we need to be sure that threads 0 - 8 have loaded the
    // message buffer table.
    __syncthreads();

    // store the data in the correct message buffer
    if ((idx < nx) && (idy < ny)) {
        //     printf("ZL PACK %d %d %d %e\n", idx, idy, idz, x);
        msgbuf[buff_index][idst] = x;
    }
}

template < typename array_t > //, typename T1, typename T2, template <typename> class T3>
void m_packZL_generic(
    array_t const &fields, typename array_t::value_type::value_type **d_msgbufTab, const int *d_msgsize)
//                      const gridtools::field_on_the_fly<T1,T2,T3> * halo_d)
{

#ifdef CUDAMSG
    // just some timing stuff
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
#endif

    // run the compression a few times, just to get a bit
    // more statistics
    int niter = fields.size();
    for (int i = 0; i < niter; i++) {

        // threads per block. Should be at least one warp in x, could be wider in y
        const int ntx = 32;
        const int nty = 8;
        const int ntz = 1;
        dim3 threads(ntx, nty, ntz);

        // form the grid to cover the entire plane. Use 1 block per z-layer
        int nx = fields[i].halos[0].s_length(-1) + fields[i].halos[0].s_length(0) + fields[i].halos[0].s_length(1);
        int ny = fields[i].halos[1].s_length(-1) + fields[i].halos[1].s_length(0) + fields[i].halos[1].s_length(1);
        int nz = fields[i].halos[2].s_length(-1);

        int nbx = (nx + ntx - 1) / ntx;
        int nby = (ny + nty - 1) / nty;
        int nbz = (nz + ntz - 1) / ntz;
        dim3 blocks(nbx, nby, nbz);

#ifdef CUDAMSG
        printf("ZL -- PACK Launch grid (%d,%d,%d) with (%d,%d,%d) threads (full size: %d,%d,%d)\n",
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
        m_packZLKernel_generic<<<blocks, threads, 0, ZL_stream>>>
        (fields[i].ptr,
         (d_msgbufTab),
         wrap_argument(d_msgsize+27*i),
         *(reinterpret_cast<const gridtools::array<gridtools::halo_descriptor,3>*>(&fields[i])),
         nx,
         ny,
         0);
// clang-format on
#ifdef CUDAMSG
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("KLF in %s : %s\n", __FILE__, cudaGetErrorString(err));
                exit(-1);
            }
#endif
        }
    }

// more timing stuff and conversion into reasonable units
// for display
#ifdef CUDAMSG
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // double nnumb =  niter * (double) (nx * ny * nz);
    // double nbyte =  nnumb * sizeof(double);

    // printf("ZL Packed %g numbers in %g ms, BW = %g GB/s\n",
    //     nnumb, elapsedTime, (nbyte/(elapsedTime/1e3))/1e9);

    printf("ZL Packed numbers in %g ms\n", elapsedTime);
#endif
}
