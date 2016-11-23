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
template < typename value_type >
__global__ void m_unpackZLKernel(value_type *__restrict__ d_data,
    value_type **__restrict__ d_msgbufTab_r,
    int *d_msgsize_r,
    const gridtools::halo_descriptor *halo /*_g*/,
    int const nx,
    int const ny,
    int const tranlation_const,
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
        msgbuf[threadIdx.x] = d_msgbufTab_r[threadIdx.x];
    }

    int aa = halo[0].minus() + halo[0].end() - halo[0].begin();
    int ab = halo[1].minus() + halo[1].end() - halo[1].begin();

    int pas = halo[0].minus();
    if (idx < halo[0].minus())
        pas = 0;

    int pbs = halo[1].minus();
    if (idy < halo[1].minus())
        pbs = 0;

    int ba = 1;
    int aas = 0;
    int la = halo[0].end() - halo[0].begin() + 1;
    if (idx < halo[0].minus()) {
        ba = 0;
        la = halo[0].minus();
    }
    if (idx > aa) {
        ba = 2;
        la = halo[0].plus();
        aas = halo[0].end() - halo[0].begin() + 1;
    }

    int bb = 1;
    int abs = 0;
    int lb = halo[1].end() - halo[1].begin() + 1;
    if (idy < halo[1].minus()) {
        bb = 0;
        lb = halo[1].minus();
    }
    if (idy > ab) {
        bb = 2;
        lb = halo[1].plus();
        abs = halo[1].end() - halo[1].begin() + 1;
    }

    int bc = 0;
    int lc = halo[2].minus();

    int oa = idx - pas - aas;
    int ob = idy - pbs - abs;
    int oc = idz;

    int buff_index = ba + 3 * bb + 9 * bc;

    int isrc = oa + ob * la + oc * la * lb + field_index * d_msgsize_r[buff_index];

    __syncthreads();
    value_type x;
    // store the data in the correct message buffer
    if ((idx < nx) && (idy < ny)) {
        x = msgbuf[buff_index][isrc];
    }

    int tli = halo[0].total_length();
    int tlj = halo[1].total_length();

    int idst = idx + idy * tli + idz * tli * tlj + tranlation_const;
    if ((idx < nx) && (idy < ny)) {
        d_data[idst] = x;
    }
}

template < typename array_t, typename value_type >
void m_unpackZL(array_t const &d_data_array,
    value_type **d_msgbufTab_r,
    int d_msgsize_r[27],
    const gridtools::halo_descriptor halo[3],
    const gridtools::halo_descriptor halo_d[3]) {
    // threads per block. Should be at least one warp in x, could be wider in y
    const int ntx = 32;
    const int nty = 8;
    const int ntz = 1;
    dim3 threads(ntx, nty, ntz);

    // form the grid to cover the entire plane. Use 1 block per z-layer
    int nx = halo[0].r_length(-1) + halo[0].r_length(0) + halo[0].r_length(1);
    int ny = halo[1].r_length(-1) + halo[1].r_length(0) + halo[1].r_length(1);
    int nz = halo[2].r_length(-1);

    int nbx = (nx + ntx - 1) / ntx;
    int nby = (ny + nty - 1) / nty;
    int nbz = (nz + ntz - 1) / ntz;
    dim3 blocks(nbx, nby, nbz);

    if (nbx == 0 || nby == 0 || nbz == 0)
        return;

#ifdef CUDAMSG
    printf("UnpackZL Launch grid (%d,%d,%d) with (%d,%d) threads tot: %dx%dx%d\n", nbx, nby, nbz, ntx, nty, nx, ny, nz);

    // just some timing stuff
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
#endif

    // run the compression a few times, just to get a bit
    // more statistics
    int niter = d_data_array.size();

    for (int i = 0; i < niter; i++) {

        // the actual kernel launch
        // clang-format off
      m_unpackZLKernel< <<blocks, threads, 0, ZL_stream> >>(d_data_array[i], d_msgbufTab_r, d_msgsize_r, halo_d, nx, ny,
                                                          (halo[0].begin()-halo[0].minus())
                                                          + (halo[1].begin()-halo[1].minus())*halo[0].total_length()
                                                          + (halo[2].begin()-halo[2].minus())*halo[0].total_length() *halo[1].total_length(), i );
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

    double nnumb = niter * (double)(nx * ny * halo[2].plus());
    double nbyte = nnumb * sizeof(double);

    printf("ZL Packed %g numbers in %g ms, BW = %g GB/s\n", nnumb, elapsedTime, (nbyte / (elapsedTime / 1e3)) / 1e9);
#endif
}
