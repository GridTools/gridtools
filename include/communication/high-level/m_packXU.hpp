template <typename value_type>
__global__ void m_packXUKernel(const value_type* __restrict__ d_data, 
                               value_type** __restrict__ d_msgbufTab, int* d_msgsize,
                               const gridtools::halo_descriptor* halo/*_g*/, int const ny, int const nz, 
                               int const field_index){
 
   // per block shared buffer for storing destination buffers
   __shared__ value_type* msgbuf[27];
   //__shared__ gridtools::halo_descriptor halo[3];

   int idx = blockIdx.x;
   int idy = blockIdx.y * blockDim.y + threadIdx.y;
   int idz = blockIdx.z * blockDim.z + threadIdx.z;

   // load msg buffer table into shmem. Only the first 9 threads 
   // need to do this
   if(threadIdx.x == 0 && threadIdx.y < 27 && threadIdx.z == 0) {
    msgbuf[threadIdx.y] =  d_msgbufTab[threadIdx.y]; 
   }

   // load the data from the contiguous source buffer
   value_type x;
   if((idy < ny) && (idz < nz)) {
     int mas=halo[0].minus();

     int ia = idx + halo[0].end() - mas + 1;;
     int ib = idy + halo[1].begin();
     int ic = idz + halo[2].begin();
     int isrc  = ia + ib * halo[0].total_length() + ic * halo[0].total_length() * halo[1].total_length();
     x =  d_data[isrc]; 
   }

   int ba = 2;
   int la = halo[0].minus(); 
     
   int bb = 1;
   int lb = halo[1].end() - halo[1].begin() + 1;

   int bc = 1;

   int b_ind = ba + 3*bb + 9*bc;

   int oa = idx;
   int ob = idy;
   int oc = idz;

   int idst = oa + ob * la + oc * la * lb + field_index * d_msgsize[b_ind];;

   // at this point we need to be sure that threads 0 - 8 have loaded the
   // message buffer table. 
   __syncthreads();
  
    // store the data in the correct message buffer
   if((idy < ny) && (idz < nz)) {
     //printf("XU %d %d %d -> %16.16e\n", idx, idy, idz, x);
     msgbuf[b_ind][idst] = x;
   }
}

template <typename array_t, typename value_type>
void m_packXU(array_t const& d_data_array, value_type** d_msgbufTab, int d_msgsize[27],
            const gridtools::halo_descriptor halo[3],
            const gridtools::halo_descriptor halo_d[3])
{
  // threads per block. Should be at least one warp in x, could be wider in y
  const int ntx = 1;                 
  const int nty = 32;
  const int ntz = 8;
  dim3 threads(ntx, nty, ntz);

  // form the grid to cover the entire plane. Use 1 block per z-layer
  int nx = halo[0].s_length(1);
  int ny = halo[1].s_length(0);
  int nz = halo[2].s_length(0);

  int nbx = (nx + ntx - 1) / ntx ;
  int nby = (ny + nty - 1) / nty ;
  int nbz = (nz + ntz - 1) / ntz ;
  dim3 blocks(nbx, nby, nbz); 

  if (nbx==0 || nby==0 || nbz==0) 
    return;

#ifdef CUDAMSG
  printf("Launch grid (%d,%d,%d) with (%d,%d,%d) threads (full size: %d,%d,%d)\n", 
         nbx, nby, nbz, ntx, nty, ntz, nx, ny, nz); 
// just some timing stuff
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
#endif 

  const int niter = d_data_array.size();

  // run the compression a few times, just to get a bit
  // more statistics
  for(int i=0; i < niter; i++){

        // the actual kernel launch
        // clang-format off
      m_packXUKernel<<<blocks, threads, 0, XU_stream>>>(d_data_array[i], d_msgbufTab, d_msgsize, halo_d, ny, nz, i);
// clang-format on
#ifdef CUDAMSG
    int err = cudaGetLastError();
    if(err != cudaSuccess){
      printf("Kernel launch failure\n");
      exit(-1);
    } 
#endif
  }

#ifdef CUDAMSG
  // more timing stuff and conversion into reasonable units
  // for display
  cudaEventRecord(stop,  0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
 
  double nnumb =  niter * (double) (nx * ny * nz); 
  double nbyte =  nnumb * sizeof(double);
 
  printf("XU Packed %g numbers in %g ms, BW = %g GB/s\n", 
      nnumb, elapsedTime, (nbyte/(elapsedTime/1e3))/1e9);
#endif
} 

