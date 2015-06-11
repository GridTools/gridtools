#include "cuda.h"
#include <mpi.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <communication/halo_exchange.h>
#include <string>
#include <stdlib.h>
#include <common/layout_map.h>
#include <common/boollist.h>
#include <sys/time.h>

int pid;
int nprocs;
MPI_Comm CartComm;
int dims[3] = {0,0,0};
int coords[3]={0,0,0};

struct timeval start_tv;
struct timeval stop1_tv;
struct timeval stop2_tv;
struct timeval stop3_tv;
double lapse_time1;
double lapse_time2;
double lapse_time3;

#define B_ADD 10000
#define C_ADD 20000

typedef gridtools::gcl_gpu arch_type;

#define MIN(a,b) ((a)<(b)?(a):(b))

template <typename T, typename lmap>
struct array {
  T *ptr;
  int n,m,l;

  array(T* _p, int _n, int _m, int _l)
    : ptr(_p)
    , n(lmap::template find<2>(_n,_m,_l))
    , m(lmap::template find<1>(_n,_m,_l))
    , l(lmap::template find<0>(_n,_m,_l))
  {}

  T &operator()(int i, int j, int k) {
    // a[(DIM1+2*H)*(DIM2+2*H)*kk+ii*(DIM2+2*H)+jj]
    return ptr[l*m*lmap::template find<2>(i,j,k)+
               l*lmap::template find<1>(i,j,k)+
               lmap::template find<0>(i,j,k)];
  }

  T const &operator()(int i, int j, int k) const {
    return ptr[l*m*lmap::template find<2>(i,j,k)+
               l*lmap::template find<1>(i,j,k)+
               lmap::template find<0>(i,j,k)];
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

//#define USE_DOUBLE

/* This is the data type of the elements of the data arrays.
 */
#ifndef USE_DOUBLE
struct triple_t {
  int _x,_y,_z;
  __host__ __device__ triple_t(int a, int b, int c): _x(a), _y(b), _z(c) {}
  __host__ __device__ triple_t(): _x(-1), _y(-1), _z(-1) {}

  __host__ __device__ triple_t(triple_t const & t)
    : _x(t._x)
    , _y(t._y)
    , _z(t._z)
  {}

  __host__ __device__ triple_t floor() {
    int m = MIN(_x, MIN(_y,_z));

    return (m==-1)?triple_t(m,m,m):*this;
  }

  __host__ __device__ int x() const {return _x;}
  __host__ __device__ int y() const {return _y;}
  __host__ __device__ int z() const {return _z;}
};

std::ostream& operator<<(std::ostream &s, triple_t const & t) {
  return s << " ("
           << t._x << ", "
           << t._y << ", "
           << t._z << ") ";
}

bool operator==(triple_t const & a, triple_t const & b) {
  return (a._x == b._x &&
          a._y == b._y &&
          a._z == b._z);
}

bool operator!=(triple_t const & a, triple_t const & b) {
  return !(a==b);
}
#else
struct triple_t {
  double value;

  __host__ __device__ triple_t(int a, int b, int c): value(static_cast<long long int>(a)*100000000+static_cast<long long int>(b)*10000+static_cast<long long int>(c)) {}

  __host__ __device__ triple_t(): value(999999999999) {}

  __host__ __device__ triple_t(triple_t const & t)
    : value(t.value)
  {}

  __host__ __device__ triple_t floor() {
    if (x() == 9999 || y() == 9999 || z() == 9999) {
      return triple_t();
    } else {
      return *this;
    }
  }

  __host__ __device__ int x() const {
    long long int cast = static_cast<long long int>(value);
    return static_cast<int>((cast/100000000)%10000);
  }

  __host__ __device__ int y() const {
    long long int cast = static_cast<long long int>(value);
    return static_cast<int>((cast/10000)%10000);
  }

  __host__ __device__ int z() const {
    long long int cast = static_cast<long long int>(value);
    return static_cast<int>((cast)%10000);
  }

};

triple_t operator*(int a, triple_t const& b) {
  return triple_t(a*b.x(), a*b.y(), a*b.z());
}

triple_t operator+(int a, triple_t const& b) {
  return triple_t(a+b.x(), a+b.y(), a+b.z());
}

std::ostream& operator<<(std::ostream &s, triple_t const & t) {
  return s << " ("
           << t.x() << ", "
           << t.y() << ", "
           << t.z() << ") ";
}

bool operator==(triple_t const & a, triple_t const & b) {
  return (a.x() == b.x() &&
          a.y() == b.y() &&
          a.z() == b.z());
}

bool operator!=(triple_t const & a, triple_t const & b) {
  return !(a==b);
}
#endif

/* Just and utility to print values
 */
template <typename array_t>
void printbuff(std::ostream &file, array_t const & a, int d1, int d2, int d3) {
  if (d1<=6 && d2<=6 && d3<=6) {
    file << "------------\n";
    for (int kk=0; kk<d3; ++kk) {
      file << "|";
      for (int jj=0; jj<d2; ++jj) {
        for (int ii=0; ii<d1; ++ii) {
          file << a(ii,jj,kk);
        }
        file << "|\n";
      }
      file << "\n\n";
    }
    file << "------------\n\n";
  }
}


template <typename ST, int I1, int I2, int I3, bool per0, bool per1, bool per2>
void run(ST & file, int DIM1, int DIM2, int DIM3, int H1m, int H1p, int H2m, int H2p, int H3m, int H3p, triple_t *_a, triple_t *_b, triple_t *_c) {

  typedef gridtools::layout_map<I1,I2,I3> layoutmap;

  array<triple_t, layoutmap > a(_a, (DIM1+H1m+H1p),(DIM2+H2m+H2p),(DIM3+H3m+H3p));
  array<triple_t, layoutmap > b(_b, (DIM1+H1m+H1p),(DIM2+H2m+H2p),(DIM3+H3m+H3p));
  array<triple_t, layoutmap > c(_c, (DIM1+H1m+H1p),(DIM2+H2m+H2p),(DIM3+H3m+H3p));

  /* Just an initialization */
  for (int ii=0; ii<DIM1+H1m+H1p; ++ii)
    for (int jj=0; jj<DIM2+H2m+H2p; ++jj) {
      for (int kk=0; kk<DIM3+H3m+H3p; ++kk) {
        a(ii,jj,kk) = triple_t();
        b(ii,jj,kk) = triple_t();
        c(ii,jj,kk) = triple_t();
      }
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
  static const int version = gridtools::version_manual; // 0 is the usual version, 1 is the one that build the whole datatype (Only vector interface supported)
#ifdef USE_DOUBLE
  typedef gridtools::halo_exchange_dynamic_ut<layoutmap,
    gridtools::layout_map<0,1,2>, double , gridtools::MPI_3D_process_grid_t<3>, arch_type, version > pattern_type;
#else
  typedef gridtools::halo_exchange_dynamic_ut<layoutmap,
    gridtools::layout_map<0,1,2>, triple_t , gridtools::MPI_3D_process_grid_t<3>, arch_type, version > pattern_type;
#endif

  /* The pattern is now instantiated with the periodicities and the
     communicator. The periodicity of the communicator is
     irrelevant. Setting it to be periodic is the best choice, then
     GCL can deal with any periodicity easily.
  */
  pattern_type he(typename pattern_type::grid_type:: period_type(per0, per1, per2), CartComm);


  /* Next we need to describe the data arrays in terms of halo
     descriptors (see the manual). The 'order' of registration, that
     is the index written within <.> follows the logical order of the
     application. That is, 0 is associated to 'i', '1' is
     associated to 'j', '2' to 'k'.
  */
  he.template add_halo<0>(H1m, H1p, H1m, DIM1+H1m-1, DIM1+H1m+H1p);
  he.template add_halo<1>(H2m, H2p, H2m, DIM2+H2m-1, DIM2+H2m+H2p);
  he.template add_halo<2>(H3m, H3p, H3m, DIM3+H3m-1, DIM3+H3m+H3p);

  /* Pattern is set up. This must be done only once per pattern. The
     parameter must me greater or equal to the largest number of
     arrays updated in a single step.
  */
  he.setup(3);


  file << "Proc: (" << coords[0] << ", " << coords[1] << ", " << coords[2] << ")\n";


  /* Data is initialized in the inner region of size DIM1xDIM2
   */
  for (int ii=H1m; ii<DIM1+H1m; ++ii)
    for (int jj=H2m; jj<DIM2+H2m; ++jj)
      for (int kk=H3m; kk<DIM3+H3m; ++kk) {
        a(ii,jj,kk) =
          triple_t(ii-H1m+(DIM1)*coords[0],
                   jj-H2m+(DIM2)*coords[1],
                   kk-H3m+(DIM3)*coords[2]);
        b(ii,jj,kk) =
          triple_t(ii-H1m+(DIM1)*coords[0]+B_ADD,
                   jj-H2m+(DIM2)*coords[1]+B_ADD,
                   kk-H3m+(DIM3)*coords[2]+B_ADD);
        c(ii,jj,kk) =
          triple_t(ii-H1m+(DIM1)*coords[0]+C_ADD,
                   jj-H2m+(DIM2)*coords[1]+C_ADD,
                   kk-H3m+(DIM3)*coords[2]+C_ADD);
      }

  printbuff(file,a, DIM1+H1m+H1p, DIM2+H2m+H2p, DIM3+H3m+H3p);

#ifdef _GCL_GPU_
  file << "GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU \n";

  triple_t* gpu_a = 0;
  triple_t* gpu_b = 0;
  triple_t* gpu_c = 0;
  cudaError_t status;
  status = cudaMalloc( &gpu_a, (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)*sizeof(triple_t));
  if( !checkCudaStatus( status ) ) return;
  status = cudaMalloc( &gpu_b, (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)*sizeof(triple_t));
  if( !checkCudaStatus( status ) ) return;
  status = cudaMalloc( &gpu_c, (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)*sizeof(triple_t));
  if( !checkCudaStatus( status ) ) return;

  status = cudaMemcpy( gpu_a, a.ptr,
                       (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)*sizeof(triple_t),
                       cudaMemcpyHostToDevice );
  if( !checkCudaStatus( status ) ) return;

  status = cudaMemcpy( gpu_b, b.ptr,
                       (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)*sizeof(triple_t),
                       cudaMemcpyHostToDevice );
  if( !checkCudaStatus( status ) ) return;

  status = cudaMemcpy( gpu_c, c.ptr,
                       (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)*sizeof(triple_t),
                       cudaMemcpyHostToDevice );
  if( !checkCudaStatus( status ) ) return;


#ifdef USE_DOUBLE
  std::vector<double*> vect(3);
  vect[0] = reinterpret_cast<double*>(gpu_a);
  vect[1] = reinterpret_cast<double*>(gpu_b);
  vect[2] = reinterpret_cast<double*>(gpu_c);
#else
  std::vector<triple_t*> vect(3);
  vect[0] = gpu_a;
  vect[1] = gpu_b;
  vect[2] = gpu_c;
#endif
  /* This is self explanatory now
   */
  he.pack(vect);

  he.exchange();

  he.unpack(vect);

  status = cudaMemcpy( a.ptr, gpu_a,
                       (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)*sizeof(triple_t),
                       cudaMemcpyDeviceToHost );
  if( !checkCudaStatus( status ) ) return;

  status = cudaMemcpy( b.ptr, gpu_b,
                       (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)*sizeof(triple_t),
                       cudaMemcpyDeviceToHost );
  if( !checkCudaStatus( status ) ) return;

  status = cudaMemcpy( c.ptr, gpu_c,
                       (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)*sizeof(triple_t),
                       cudaMemcpyDeviceToHost );
  if( !checkCudaStatus( status ) ) return;

  status = cudaFree( gpu_a );
  if( !checkCudaStatus( status ) ) return;
  status = cudaFree( gpu_b );
  if( !checkCudaStatus( status ) ) return;
  status = cudaFree( gpu_c );
  if( !checkCudaStatus( status ) ) return;

#else
  /* This is self explanatory now
   */

  std::vector<triple_t*> vect(3);
  vect[0] = a.ptr;
  vect[1] = b.ptr;
  vect[2] = c.ptr;
  MPI_Barrier(gridtools::GCL_WORLD);

  gettimeofday(&start_tv, NULL);

  he.pack(vect);

  gettimeofday(&stop1_tv, NULL);

  he.start_exchange();
  //  MPI_Barrier(MPI_COMM_WORLD);
  he.wait();

  gettimeofday(&stop2_tv, NULL);

  he.unpack(vect);

  gettimeofday(&stop3_tv, NULL);

  lapse_time1 = ((static_cast<double>(stop1_tv.tv_sec)+1/1000000.0*static_cast<double>(stop1_tv.tv_usec)) - (static_cast<double>(start_tv.tv_sec)+1/1000000.0*static_cast<double>(start_tv.tv_usec))) * 1000.0;

  lapse_time2 = ((static_cast<double>(stop2_tv.tv_sec)+1/1000000.0*static_cast<double>(stop2_tv.tv_usec)) - (static_cast<double>(stop1_tv.tv_sec)+1/1000000.0*static_cast<double>(stop1_tv.tv_usec))) * 1000.0;

  lapse_time3 = ((static_cast<double>(stop3_tv.tv_sec)+1/1000000.0*static_cast<double>(stop3_tv.tv_usec)) - (static_cast<double>(stop2_tv.tv_sec)+1/1000000.0*static_cast<double>(stop2_tv.tv_usec))) * 1000.0;

  MPI_Barrier(MPI_COMM_WORLD);
  file << "TIME PACK: " << lapse_time1 << std::endl;
  file << "TIME EXCH: " << lapse_time2 << std::endl;
  file << "TIME UNPK: " << lapse_time3 << std::endl;
  file << "TIME ALL : " << lapse_time1+lapse_time2+lapse_time3 << std::endl;
#endif

  file << "\n********************************************************************************\n";

  printbuff(file,a, DIM1+H1m+H1p, DIM2+H2m+H2p, DIM3+H3m+H3p);

  int passed = true;


  /* Checking the data arrived correctly in the whole region
   */
  for (int ii=0; ii<DIM1+H1m+H1p; ++ii)
    for (int jj=0; jj<DIM2+H2m+H2p; ++jj)
      for (int kk=0; kk<DIM3+H3m+H3p; ++kk) {

        triple_t ta;
        triple_t tb;
        triple_t tc;
        int tax, tay, taz;
        int tbx, tby, tbz;
        int tcx, tcy, tcz;

        tax = modulus(ii-H1m+(DIM1)*coords[0], DIM1*dims[0]);
        tbx = modulus(ii-H1m+(DIM1)*coords[0], DIM1*dims[0])+B_ADD;
        tcx = modulus(ii-H1m+(DIM1)*coords[0], DIM1*dims[0])+C_ADD;

        tay = modulus(jj-H2m+(DIM2)*coords[1], DIM2*dims[1]);
        tby = modulus(jj-H2m+(DIM2)*coords[1], DIM2*dims[1])+B_ADD;
        tcy = modulus(jj-H2m+(DIM2)*coords[1], DIM2*dims[1])+C_ADD;

        taz = modulus(kk-H3m+(DIM3)*coords[2], DIM3*dims[2]);
        tbz = modulus(kk-H3m+(DIM3)*coords[2], DIM3*dims[2])+B_ADD;
        tcz = modulus(kk-H3m+(DIM3)*coords[2], DIM3*dims[2])+C_ADD;

        if (!per0) {
          if ( ((coords[0]==0) && (ii<H1m)) ||
               ((coords[0] == dims[0]-1) && (ii >= DIM1+H1m)) ) {
            tax=triple_t().x();
            tbx=triple_t().x();
            tcx=triple_t().x();
          }
        }

        if (!per1) {
          if ( ((coords[1]==0) && (jj<H2m)) ||
               ((coords[1] == dims[1]-1) && (jj >= DIM2+H2m)) ) {
            tay=triple_t().y();
            tby=triple_t().y();
            tcy=triple_t().y();
          }
        }

        if (!per2) {
          if ( ((coords[2]==0) && (kk<H3m)) ||
               ((coords[2] == dims[2]-1) && (kk >= DIM3+H3m)) ) {
            taz=triple_t().z();
            tbz=triple_t().z();
            tcz=triple_t().z();
          }
        }

        ta = triple_t(tax, tay, taz).floor();
        tb = triple_t(tbx, tby, tbz).floor();
        tc = triple_t(tcx, tcy, tcz).floor();

        if (a(ii,jj,kk) != ta) {
          passed = false;
          file << "a " << a(ii,jj,kk) << " != "
               << ta
               << "\n";
        }

        if (b(ii,jj,kk) != tb) {
          passed = false;
          file << "b " << b(ii,jj,kk) << " != "
               << tb
               << "\n";
        }

        if (c(ii,jj,kk) != tc) {
          passed = false;
          file << "c " << c(ii,jj,kk) << " != "
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
  gridtools::GCL_Init(argc, argv);

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

  MPI_Dims_create(nprocs, 3, dims);
  int period[3] = {1, 1, 1};

  file << "@" << pid << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << " - " << dims[2] << "\n";

  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, period, false, &CartComm);

  MPI_Cart_get(CartComm, 3, dims, period, coords);


  /* Each process will hold a tile of size
     (DIM1+2*H)x(DIM2+2*H)x(DIM3+2*H). The DIM1xDIM2xDIM3 area inside
     the H width border is the inner region of an hypothetical stencil
     computation whise halo width is H.
   */
  int DIM1=atoi(argv[1]);
  int DIM2=atoi(argv[2]);
  int DIM3=atoi(argv[3]);
  int H1m =atoi(argv[4]);
  int H1p =atoi(argv[5]);
  int H2m =atoi(argv[6]);
  int H2p =atoi(argv[7]);
  int H3m =atoi(argv[8]);
  int H3p =atoi(argv[9]);


  /* This example will exchange 3 data arrays at the same time with
     different values.
   */
  triple_t *_a = new triple_t[(DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)];
  triple_t *_b = new triple_t[(DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)];
  triple_t *_c = new triple_t[(DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)];


  file << "Permutation 0,1,2\n";

  file << "run<std::ostream, 0,1,2, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 0,1,2, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 0,1,2, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 0,1,2, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 0,1,2, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 0,1,2, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 0,1,2, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 0,1,2, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 0,1,2, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 0,1,2, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 0,1,2, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 0,1,2, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 0,1,2, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 0,1,2, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 0,1,2, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 0,1,2, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);
  file << "---------------------------------------------------\n";


  file << "Permutation 0,2,1\n";

  file << "run<std::ostream, 0,2,1, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 0,2,1, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 0,2,1, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 0,2,1, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 0,2,1, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 0,2,1, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 0,2,1, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 0,2,1, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 0,2,1, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 0,2,1, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 0,2,1, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 0,2,1, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 0,2,1, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 0,2,1, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 0,2,1, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 0,2,1, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);
  file << "---------------------------------------------------\n";


  file << "Permutation 1,0,2\n";

  file << "run<std::ostream, 1,0,2, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 1,0,2, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 1,0,2, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 1,0,2, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 1,0,2, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 1,0,2, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 1,0,2, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 1,0,2, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 1,0,2, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 1,0,2, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 1,0,2, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 1,0,2, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 1,0,2, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 1,0,2, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 1,0,2, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 1,0,2, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);
  file << "---------------------------------------------------\n";


  file << "Permutation 1,2,0\n";

  file << "run<std::ostream, 1,2,0, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 1,2,0, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 1,2,0, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 1,2,0, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 1,2,0, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 1,2,0, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 1,2,0, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 1,2,0, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 1,2,0, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 1,2,0, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 1,2,0, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 1,2,0, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 1,2,0, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 1,2,0, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 1,2,0, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 1,2,0, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);
  file << "---------------------------------------------------\n";


  file << "Permutation 2,0,1\n";

  file << "run<std::ostream, 2,0,1, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 2,0,1, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 2,0,1, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 2,0,1, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 2,0,1, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 2,0,1, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 2,0,1, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 2,0,1, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 2,0,1, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 2,0,1, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 2,0,1, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 2,0,1, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 2,0,1, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 2,0,1, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 2,0,1, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 2,0,1, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);
  file << "---------------------------------------------------\n";


  file << "Permutation 2,1,0\n";

  file << "run<std::ostream, 2,1,0, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 2,1,0, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 2,1,0, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 2,1,0, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 2,1,0, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 2,1,0, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 2,1,0, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 2,1,0, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 2,1,0, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 2,1,0, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 2,1,0, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 2,1,0, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 2,1,0, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 2,1,0, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);

  file << "run<std::ostream, 2,1,0, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c)\n";
  run<std::ostream, 2,1,0, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, _b, _c);
  file << "---------------------------------------------------\n";


  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
