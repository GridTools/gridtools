
/*
Copyright (c) 2012, MAURO BIANCO, UGO VARETTO, SWISS NATIONAL SUPERCOMPUTING CENTRE (CSCS)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Swiss National Supercomputing Centre (CSCS) nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL MAURO BIANCO, UGO VARETTO, OR 
SWISS NATIONAL SUPERCOMPUTING CENTRE (CSCS), BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <mpi.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <halo_exchange.h>
#include <string>
#include <stdlib.h>
#include <utils/layout_map.h>
#include <utils/boollist.h>

int pid;
int nprocs;
MPI_Comm CartComm;
int dims[2] = {0,0};
int coords[2]={0,0};

#ifdef _GCL_GPU_
typedef GCL::gcl_gpu arch_type;
#else
typedef GCL::gcl_cpu arch_type;
#endif

#define VECTOR_INT
#ifdef VECTOR_INT
typedef int type1;
typedef int type2;
typedef int type3;
#else
typedef int type1;
typedef float type2;
typedef double type3;
#endif

template <typename T, typename lmap>
struct array {
  T *ptr;
  int n,m;

  array(T* _p, int _n, int _m)
    : ptr(_p)
    , n(lmap::template find<1>(_n,_m))
    , m(lmap::template find<0>(_n,_m))  
  {}

  T& operator()(int i, int j) {
    return ptr[m*lmap::template find<1>(i,j)+
               lmap::template find<0>(i,j)];
  }

  T const& operator()(int i, int j) const {
    return ptr[m*lmap::template find<1>(i,j)+
               lmap::template find<0>(i,j)];
  }

  operator void*() const {return reinterpret_cast<void*>(ptr);}
  operator T*() const {return ptr;}
};

/** \file Example of use of halo_exchange pattern for regular
    grids. The comments in the code aim at highlight the process of
    instantiating and running a halo exchange pattern.
*/

inline int modulus(int __i, int __j) {
  return (((((__i%__j)<0)?(__j+__i%__j):(__i%__j))));
}

/* This is the data type of the elements of the data arrays.
 */
template <typename T>
struct pair_t {
  T x,y;
  pair_t(T a, T b): x(a), y(b) {}
  pair_t(): x(-1), y(-1) {}

  pair_t(pair_t const & t)
    : x(t.x)
    , y(t.y)
  {}

  pair_t<T> floor() {
    T m = std::min(x, y);
    
    return (m==-1)?pair_t<T>(m,m):*this;
  }

};

template <typename T>
std::ostream& operator<<(std::ostream &s, pair_t<T> const & t) { 
  return s << " (" 
           << t.x << ", "
           << t.y << ") ";
}

template <typename T>
bool operator==(pair_t<T> const & a, pair_t<T> const & b) {
  return (a.x == b.x && 
          a.y == b.y);
}

template <typename T>
bool operator!=(pair_t<T> const & a, pair_t<T> const & b) {
  return !(a==b);
}

/* Just and utility to print values
 */
template <typename array_t>
void printbuff(std::ostream &file, array_t const & a, int d1, int d2) {
  if (d1<=6 && d2<=6) {
    for (int jj=0; jj<d2; ++jj) {
      file << "| ";
      for (int ii=0; ii<d1; ++ii) {
        file << a(ii,jj);
      }
      file << "|\n";
    }
    file << "\n\n";
  }
}


template <typename ST, int I1, int I2, bool per0, bool per1>
void run(ST & file, int DIM1, int DIM2, int H, pair_t<type1> *_a, pair_t<type2> *_b, pair_t<type3> *_c) {

  typedef GCL::layout_map<I1,I2> layoutmap;
  
  array<pair_t<type1>,    layoutmap > a(_a, (DIM1+2*H),(DIM2+2*H));
  array<pair_t<type2>,  layoutmap > b(_b, (DIM1+2*H),(DIM2+2*H));
  array<pair_t<type3>, layoutmap > c(_c, (DIM1+2*H),(DIM2+2*H));

  /* Just an initialization */
  for (int ii=0; ii<DIM1+2*H; ++ii)
    for (int jj=0; jj<DIM2+2*H; ++jj) {
      a(ii,jj) = pair_t<type1>();
      b(ii,jj) = pair_t<type2>();                                      
      c(ii,jj) = pair_t<type3>();
    }



  /* The pattern type is defined with the layouts, data types and
     number of dimensions.

     The logical assumption done in the program is that 'i' is the
     first dimension (rows), 'j' is the second, and 'k' is the
     third. The first layout states that 'i' is the second dimension
     in order of strides, while 'j' is the first and 'k' is the third
     (just by looking at the initialization loops this shoule be
     clear).

     The second layout states that the first dimension in data ('i')
     identify also the first dimension in the communicator. Logically,
     moving on 'i' dimension from processot (p,q,r) will lead you
     logically to processor (p+1,q,r). The other dimensions goes as
     the others.
   */
  typedef GCL::halo_exchange_generic<GCL::layout_map<0,1>, 2, arch_type > pattern_type;


  /* The pattern is now instantiated with the periodicities and the
     communicator. The periodicity of the communicator is
     irrelevant. Setting it to be periodic is the best choice, then
     GCL can deal with any periodicity easily.
  */
  pattern_type he(typename pattern_type::grid_type::period_type(per0, per1), CartComm);


  GCL::array<GCL::halo_descriptor,2> halo_dsc;
  halo_dsc[0] = GCL::halo_descriptor(H, H, H, DIM1+H-1, DIM1+2*H);
  halo_dsc[1] = GCL::halo_descriptor(H, H, H, DIM2+H-1, DIM2+2*H);
  GCL::field_on_the_fly<pair_t<type1>, layoutmap, pattern_type::traits> field1(a.ptr, halo_dsc);
  GCL::field_on_the_fly<pair_t<type2>, layoutmap, pattern_type::traits> field2(b.ptr, halo_dsc);
  GCL::field_on_the_fly<pair_t<type3>, layoutmap, pattern_type::traits> field3(c.ptr, halo_dsc);


  /* Pattern is set up. This must be done only once per pattern. The
     parameter must me greater or equal to the largest number of
     arrays updated in a single step.
  */
  //he.setup(100, halo_dsc, sizeof(double));
  he.setup(3, GCL::field_on_the_fly<int,layoutmap,pattern_type::traits>(NULL,halo_dsc), std::max(sizeof(pair_t<type1>), std::max(sizeof(pair_t<type2>),sizeof(pair_t<type3>)))); // Estimates the size


  file << "Proc: (" << coords[0] << ", " << coords[1] << ")\n";


  /* Data is initialized in the inner region of size DIM1xDIM2
   */
  for (int ii=H; ii<DIM1+H; ++ii)
    for (int jj=H; jj<DIM2+H; ++jj) {
      a(ii,jj) = 
        pair_t<type1>(ii-H+(DIM1)*coords[0],
                      jj-H+(DIM2)*coords[1]);
      b(ii,jj) = 
        pair_t<type2>(ii-H+(DIM1)*coords[0]+1,
                      jj-H+(DIM2)*coords[1]+1);
      c(ii,jj) = 
        pair_t<type3>(ii-H+(DIM1)*coords[0]+100,
                      jj-H+(DIM2)*coords[1]+100);
    }

  printbuff(file,a, DIM1+2*H, DIM2+2*H);
  //  printbuff(file,b, DIM1+2*H, DIM2+2*H, DIM3+2*H);
  //  printbuff(file,c, DIM1+2*H, DIM2+2*H, DIM3+2*H);
  
#ifdef _GCL_GPU_
  file << "GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU \n";

  pair_t<type1>* gpu_a = 0;
  pair_t<type2>* gpu_b = 0;
  pair_t<type3>* gpu_c = 0;
  cudaError_t status;
  status = cudaMalloc( &gpu_a, (DIM1+2*H)*(DIM2+2*H)*sizeof(pair_t<type1>));
  if( !checkCudaStatus( status ) ) return;
  status = cudaMalloc( &gpu_b, (DIM1+2*H)*(DIM2+2*H)*sizeof(pair_t<type2>));
  if( !checkCudaStatus( status ) ) return;
  status = cudaMalloc( &gpu_c, (DIM1+2*H)*(DIM2+2*H)*sizeof(pair_t<type3>));
  if( !checkCudaStatus( status ) ) return;

  status = cudaMemcpy( gpu_a, a.ptr,
                       (DIM1+2*H)*(DIM2+2*H)*sizeof(pair_t<type1>), 
                       cudaMemcpyHostToDevice );
  if( !checkCudaStatus( status ) ) return;

  status = cudaMemcpy( gpu_b, b.ptr,
                       (DIM1+2*H)*(DIM2+2*H)*sizeof(pair_t<type2>), 
                       cudaMemcpyHostToDevice );
  if( !checkCudaStatus( status ) ) return;

  status = cudaMemcpy( gpu_c, c.ptr,
                       (DIM1+2*H)*(DIM2+2*H)*sizeof(pair_t<type3>), 
                       cudaMemcpyHostToDevice );
  if( !checkCudaStatus( status ) ) return;


  GCL::field_on_the_fly<pair_t<type1>, layoutmap, pattern_type::traits> field1_gpu(gpu_a, halo_dsc);
  GCL::field_on_the_fly<pair_t<type2>, layoutmap, pattern_type::traits> field2_gpu(gpu_b, halo_dsc);
  GCL::field_on_the_fly<pair_t<type3>, layoutmap, pattern_type::traits> field3_gpu(gpu_c, halo_dsc);

#ifdef VECTOR_INT
  std::vector<GCL::field_on_the_fly<pair_t<type1>, layoutmap, pattern_type::traits> > vect(3);
  vect[0] = field1_gpu;
  vect[1] = field2_gpu;
  vect[2] = field3_gpu;


  he.pack(vect);

  he.exchange();

  he.unpack(vect);
#else
  /* This is self explanatory now
   */
  he.pack(field1_gpu,field2_gpu,field3_gpu);

  he.exchange();

  he.unpack(field1_gpu,field2_gpu,field3_gpu);
#endif

  status = cudaMemcpy( a.ptr, gpu_a,
                       (DIM1+2*H)*(DIM2+2*H)*sizeof(pair_t<type1>),
                       cudaMemcpyDeviceToHost );
  if( !checkCudaStatus( status ) ) return;

  status = cudaMemcpy( b.ptr, gpu_b,
                       (DIM1+2*H)*(DIM2+2*H)*sizeof(pair_t<type2>),
                       cudaMemcpyDeviceToHost );
  if( !checkCudaStatus( status ) ) return;

  status = cudaMemcpy( c.ptr, gpu_c,
                       (DIM1+2*H)*(DIM2+2*H)*sizeof(pair_t<type3>),
                       cudaMemcpyDeviceToHost );
  if( !checkCudaStatus( status ) ) return;

  status = cudaFree( gpu_a );
  if( !checkCudaStatus( status ) ) return;
  status = cudaFree( gpu_b );
  if( !checkCudaStatus( status ) ) return;
  status = cudaFree( gpu_c );
  if( !checkCudaStatus( status ) ) return;

#else

#ifdef VECTOR_INT
  std::vector<GCL::field_on_the_fly<pair_t<type1>, layoutmap, pattern_type::traits> > vect(3);
  vect[0] = field1;
  vect[1] = field2;
  vect[2] = field3;


  he.pack(vect);

  he.exchange();

  he.unpack(vect);
#else
  /* This is self explanatory now
   */
  he.pack(field1,field2,field3);

  he.exchange();

  he.unpack(field1,field2,field3);
#endif

#endif

  file << "\n********************************************************************************\n";

  printbuff(file,a, DIM1+2*H, DIM2+2*H);
  //  printbuff(file,b, DIM1+2*H, DIM2+2*H, DIM3+2*H);
  //  printbuff(file,c, DIM1+2*H, DIM2+2*H, DIM3+2*H);

  int passed = true;


  /* Checking the data arrived correctly in the whole region
   */
  for (int ii=0; ii<DIM1+2*H; ++ii)
    for (int jj=0; jj<DIM2+2*H; ++jj) {

      pair_t<type1> ta;
      pair_t<type2> tb;
      pair_t<type3> tc;
      int tax, tay;
      int tbx, tby;
      int tcx, tcy;

      tax = modulus(ii-H+(DIM1)*coords[0], DIM1*dims[0]);
      tbx = modulus(ii-H+(DIM1)*coords[0], DIM1*dims[0])+1;
      tcx = modulus(ii-H+(DIM1)*coords[0], DIM1*dims[0])+100;

      tay = modulus(jj-H+(DIM2)*coords[1], DIM2*dims[1]);
      tby = modulus(jj-H+(DIM2)*coords[1], DIM2*dims[1])+1;
      tcy = modulus(jj-H+(DIM2)*coords[1], DIM2*dims[1])+100;

      if (!per0) {
        if ( ((coords[0]==0) && (ii<H)) || 
             ((coords[0] == dims[0]-1) && (ii >= DIM1+H)) ) {
          tax=pair_t<type1>().x;
          tbx=pair_t<type2>().x;
          tcx=pair_t<type3>().x;
        }
      }

      if (!per1) {
        if ( ((coords[1]==0) && (jj<H)) || 
             ((coords[1] == dims[1]-1) && (jj >= DIM2+H)) ) {
          tay=pair_t<type1>().y;
          tby=pair_t<type2>().y;
          tcy=pair_t<type3>().y;
        }
      }

      ta = pair_t<type1>(tax, tay).floor();
      tb = pair_t<type2>(tbx, tby).floor();
      tc = pair_t<type3>(tcx, tcy).floor();

      if (a(ii,jj) != ta) {
        passed = false;
        file << "a " << a(ii,jj) << " != " 
             << ta
             << "\n";
      }

      if (b(ii,jj) != tb) {
        passed = false;
        file << "b " << b(ii,jj) << " != " 
             << tb
             << "\n";
      }

      if (c(ii,jj) != tc) {
        passed = false;
        file << "c " << c(ii,jj) << " != " 
             << tc
             << "\n";
      }
    }

  if (passed)
    file << "RESULT: PASSED!\n";
  else
    file << "RESULT: FAILED!\n";
}

#ifdef _GCL_GPU_
/* device_binding added by Devendar Bureddy, OSU */

void
device_binding ()
{

  int local_rank=0/*, num_local_procs*/;
  int dev_count, use_dev_count, my_dev_id;
  char *str;

  if ((str = getenv ("MV2_COMM_WORLD_LOCAL_RANK")) != NULL)
    {
      local_rank = atoi (str);
      printf ("MV2_COMM_WORLD_LOCAL_RANK %s\n", str);
    }

  if ((str = getenv ("MPISPAWN_LOCAL_NPROCS")) != NULL)
    {
      //num_local_procs = atoi (str);
      printf ("MPISPAWN_LOCAL_NPROCS %s\n", str);
    }

  cudaGetDeviceCount (&dev_count);
  if ((str = getenv ("NUM_GPU_DEVICES")) != NULL)
    {
      use_dev_count = atoi (str);
      printf ("NUM_GPU_DEVICES %s\n", str);
    }
  else
    {
      use_dev_count = dev_count;
    }

  my_dev_id = local_rank % use_dev_count;
  printf ("local rank = %d dev id = %d\n", local_rank, my_dev_id);
  cudaSetDevice (my_dev_id);
}
#endif

int main(int argc, char** argv) {

#ifdef _GCL_GPU_
  device_binding();
#endif

  /* this example is based on MPI Cart Communicators, so we need to
  initialize MPI. This can be done by GCL automatically
  */
  MPI_Init(&argc, &argv);

  /* Now let us initialize GCL itself. If MPI is not initialized at
     this point, it will initialize it
   */
  GCL::GCL_Init(argc, argv);

  /* Here we compute the computing gris as in many applications
   */
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  std::cout << pid << " " << nprocs << "\n";

  std::stringstream ss;
  ss << pid;

  std::string filename = "out" + ss.str() + ".txt";

  std::cout << filename << std::endl;
  std::ofstream file(filename.c_str());

  file << pid << "  " << nprocs << "\n";

  MPI_Dims_create(nprocs, 2, dims);
  int period[3] = {1, 1};

  file << "@" << pid << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << "\n";
 
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, false, &CartComm);

  MPI_Cart_get(CartComm, 2, dims, period, coords);


  /* Each process will hold a tile of size
     (DIM1+2*H)x(DIM2+2*H)x(DIM3+2*H). The DIM1xDIM2xDIM3 area inside
     the H width border is the inner region of an hypothetical stencil
     computation whise halo width is H.
   */
  int DIM1=atoi(argv[1]);
  int DIM2=atoi(argv[2]);
  int H   =atoi(argv[3]);

  /* This example will exchange 3 data arrays at the same time with
     different values.
   */
  pair_t<type1> *_a = new pair_t<type1>[(DIM1+2*H)*(DIM2+2*H)];
  pair_t<type2> *_b = new pair_t<type2>[(DIM1+2*H)*(DIM2+2*H)];
  pair_t<type3> *_c = new pair_t<type3>[(DIM1+2*H)*(DIM2+2*H)];


  file << "Permutation 0,1\n";

  file << "run<std::ostream, 0,1, true, true>(file, DIM1, DIM2, H, _a, _b, _C)\n";
  run<std::ostream, 0,1, true, true>(file, DIM1, DIM2, H, _a, _b, _c);
  file.flush();

  file << "run<std::ostream, 1,0, true, true>(file, DIM1, DIM2, H, _a, _b, _C)\n";
  run<std::ostream, 1,0, true, true>(file, DIM1, DIM2, H, _a, _b, _c);
  file.flush();

  file << "run<std::ostream, 0,1, true, false>(file, DIM1, DIM2, H, _a, _b, _C)\n";
  run<std::ostream, 0,1, true, false>(file, DIM1, DIM2, H, _a, _b, _c);
  file.flush();

  file << "run<std::ostream, 1,0, true, false>(file, DIM1, DIM2, H, _a, _b, _c)\n";
  run<std::ostream, 1,0, true, false>(file, DIM1, DIM2, H, _a, _b, _c);
  file.flush();

  file << "run<std::ostream, 0,1, false, true>(file, DIM1, DIM2, H, _a, _b, _c)\n";
  run<std::ostream, 0,1, false, true>(file, DIM1, DIM2, H, _a, _b, _c);
  file.flush();

  file << "run<std::ostream, 1,0, false, true>(file, DIM1, DIM2, H, _a, _b, _c)\n";
  run<std::ostream, 1,0, false, true>(file, DIM1, DIM2, H, _a, _b, _c);
  file.flush();

  file << "run<std::ostream, 0,1, false, false>(file, DIM1, DIM2, H, _a, _b, _c)\n";
  run<std::ostream, 0,1, false, false>(file, DIM1, DIM2, H, _a, _b, _c);
  file.flush();

  file << "run<std::ostream, 1,0, false, false>(file, DIM1, DIM2, H, _a, _b, _c)\n";
  run<std::ostream, 1,0, false, false>(file, DIM1, DIM2, H, _a, _b, _c);
  file.flush();

  file << "---------------------------------------------------\n";


  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
