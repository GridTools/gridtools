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

#include "triplet.h"

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
double lapse_time4;

#define B_ADD 1
#define C_ADD 2

typedef gridtools::gcl_gpu arch_type;


template <typename ST, int I1, int I2, int I3, bool per0, bool per1, bool per2>
void run(ST & file, int DIM1, int DIM2, int DIM3, int H1m, int H1p, int H2m, int H2p, int H3m, int H3p, triple_t<USE_DOUBLE> *_a, triple_t<USE_DOUBLE> *_b, triple_t<USE_DOUBLE> *_c) {

  typedef gridtools::layout_map<I1,I2,I3> layoutmap;
  
  array<triple_t<USE_DOUBLE>, layoutmap > a(_a, (DIM1+H1m+H1p),(DIM2+H2m+H2p),(DIM3+H3m+H3p));
  array<triple_t<USE_DOUBLE>, layoutmap > b(_b, (DIM1+H1m+H1p),(DIM2+H2m+H2p),(DIM3+H3m+H3p));
  array<triple_t<USE_DOUBLE>, layoutmap > c(_c, (DIM1+H1m+H1p),(DIM2+H2m+H2p),(DIM3+H3m+H3p));

  /* Just an initialization */
  for (int ii=0; ii<DIM1+H1m+H1p; ++ii)
    for (int jj=0; jj<DIM2+H2m+H2p; ++jj) {
      for (int kk=0; kk<DIM3+H3m+H3p; ++kk) {
        a(ii,jj,kk) = triple_t<USE_DOUBLE>();
        b(ii,jj,kk) = triple_t<USE_DOUBLE>();                                      
        c(ii,jj,kk) = triple_t<USE_DOUBLE>();
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
  typedef gridtools::halo_exchange_generic<gridtools::layout_map<0,1,2>, 3, arch_type, gridtools::version_manual > pattern_type;


  /* The pattern is now instantiated with the periodicities and the
     communicator. The periodicity of the communicator is
     irrelevant. Setting it to be periodic is the best choice, then
     GCL can deal with any periodicity easily.
  */
  pattern_type he(typename pattern_type::grid_type::period_type(per0, per1, per2), CartComm);


  gridtools::array<gridtools::halo_descriptor,3> halo_dsc;
  halo_dsc[0] = gridtools::halo_descriptor(H1m, H1p, H1m, DIM1+H1m-1, DIM1+H1m+H1p);
  halo_dsc[1] = gridtools::halo_descriptor(H2m, H2p, H2m, DIM2+H2m-1, DIM2+H2m+H2p);
  halo_dsc[2] = gridtools::halo_descriptor(H3m, H3p, H3m, DIM3+H3m-1, DIM3+H3m+H3p);

  gridtools::field_on_the_fly<triple_t<USE_DOUBLE>::data_type, layoutmap, pattern_type::traits> 
      field1(reinterpret_cast<triple_t<USE_DOUBLE>::data_type*>(a.ptr), halo_dsc);
  gridtools::field_on_the_fly<triple_t<USE_DOUBLE>::data_type, layoutmap, pattern_type::traits> 
      field2(reinterpret_cast<triple_t<USE_DOUBLE>::data_type*>(b.ptr), halo_dsc);
  gridtools::field_on_the_fly<triple_t<USE_DOUBLE>::data_type, layoutmap, pattern_type::traits> 
      field3(reinterpret_cast<triple_t<USE_DOUBLE>::data_type*>(c.ptr), halo_dsc);

  /* Pattern is set up. This must be done only once per pattern. The
     parameter must me greater or equal to the largest number of
     arrays updated in a single step.
  */
  //he.setup(100, halo_dsc, sizeof(double));
  he.setup(3, gridtools::field_on_the_fly<int,layoutmap, pattern_type::traits>(NULL,halo_dsc), sizeof(triple_t<USE_DOUBLE>)); // Estimates the size

  file << "Proc: (" << coords[0] << ", " << coords[1] << ", " << coords[2] << ")\n";


  for (int ii=H1m; ii<DIM1+H1m; ++ii)
    for (int jj=H2m; jj<DIM2+H2m; ++jj) 
      for (int kk=H3m; kk<DIM3+H3m; ++kk) {
        a(ii,jj,kk) = 
          triple_t<USE_DOUBLE>(ii-H1m+(DIM1)*coords[0],
                   jj-H2m+(DIM2)*coords[1],
                   kk-H3m+(DIM3)*coords[2]);
        b(ii,jj,kk) = 
          triple_t<USE_DOUBLE>(ii-H1m+(DIM1)*coords[0]+B_ADD,
                   jj-H2m+(DIM2)*coords[1]+B_ADD,
                   kk-H3m+(DIM3)*coords[2]+B_ADD);
        c(ii,jj,kk) = 
          triple_t<USE_DOUBLE>(ii-H1m+(DIM1)*coords[0]+C_ADD,
                   jj-H2m+(DIM2)*coords[1]+C_ADD,
                   kk-H3m+(DIM3)*coords[2]+C_ADD);
      }

  file << "A \n";
  printbuff(file,a, DIM1+H1m+H1p, DIM2+H2m+H2p, DIM3+H3m+H3p);
  file << "B \n";
  printbuff(file,b, DIM1+H1m+H1p, DIM2+H2m+H2p, DIM3+H3m+H3p);
  file << "C \n";
  printbuff(file,c, DIM1+H1m+H1p, DIM2+H2m+H2p, DIM3+H3m+H3p);
  file.flush();

  file << "GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU GPU \n";

  triple_t<USE_DOUBLE>::data_type* gpu_a = 0;
  triple_t<USE_DOUBLE>::data_type* gpu_b = 0;
  triple_t<USE_DOUBLE>::data_type* gpu_c = 0;
  cudaError_t status;
  status = cudaMalloc( &gpu_a, (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)
                       *sizeof(triple_t<USE_DOUBLE>::data_type));
  if( !checkCudaStatus( status ) ) return;
  status = cudaMalloc( &gpu_b, (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)
                       *sizeof(triple_t<USE_DOUBLE>::data_type));
  if( !checkCudaStatus( status ) ) return;
  status = cudaMalloc( &gpu_c, (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)
                       *sizeof(triple_t<USE_DOUBLE>::data_type));
  if( !checkCudaStatus( status ) ) return;

  status = cudaMemcpy( gpu_a, a.ptr,
                       (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)
                       *sizeof(triple_t<USE_DOUBLE>::data_type), 
                       cudaMemcpyHostToDevice );
  if( !checkCudaStatus( status ) ) return;

  status = cudaMemcpy( gpu_b, b.ptr,
                       (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)
                       *sizeof(triple_t<USE_DOUBLE>::data_type), 
                       cudaMemcpyHostToDevice );
  if( !checkCudaStatus( status ) ) return;

  status = cudaMemcpy( gpu_c, c.ptr,
                       (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)
                       *sizeof(triple_t<USE_DOUBLE>::data_type), 
                       cudaMemcpyHostToDevice );
  if( !checkCudaStatus( status ) ) return;


  gridtools::field_on_the_fly<triple_t<USE_DOUBLE>::data_type, layoutmap, pattern_type::traits> 
      field1_gpu(gpu_a, halo_dsc);
  gridtools::field_on_the_fly<triple_t<USE_DOUBLE>::data_type, layoutmap, pattern_type::traits> 
      field2_gpu(gpu_b, halo_dsc);
  gridtools::field_on_the_fly<triple_t<USE_DOUBLE>::data_type, layoutmap, pattern_type::traits> 
      field3_gpu(gpu_c, halo_dsc);
  std::vector<gridtools::field_on_the_fly<triple_t<USE_DOUBLE>, layoutmap, pattern_type::traits> > vect(3);

  //#define VECTOR_INTERFACE
#ifdef VECTOR_INTERFACE
  vect[0] = field1_gpu;
  vect[1] = field2_gpu;
  vect[2] = field3_gpu;
  // std::vector<triple_t<USE_DOUBLE>*> vect(3);
  // vect[0] = gpu_a;
  // vect[1] = gpu_b;
  // vect[2] = gpu_c;
  // /* This is self explanatory now

  MPI_Barrier(MPI_COMM_WORLD);

  gettimeofday(&start_tv, NULL);
  he.pack(vect);

  gettimeofday(&stop1_tv, NULL);
  he.exchange();

  gettimeofday(&stop2_tv, NULL);
  he.unpack(vect);

  gettimeofday(&stop3_tv, NULL);
#else
  MPI_Barrier(MPI_COMM_WORLD);

  gettimeofday(&start_tv, NULL);
  he.pack(field1_gpu, field2_gpu, field3_gpu);

  gettimeofday(&stop1_tv, NULL);
  he.exchange();

  gettimeofday(&stop2_tv, NULL);
  he.unpack(field1_gpu, field2_gpu, field3_gpu);

  gettimeofday(&stop3_tv, NULL);
#endif

  lapse_time1 = ((static_cast<double>(stop1_tv.tv_sec)+1/1000000.0*static_cast<double>(stop1_tv.tv_usec)) - (static_cast<double>(start_tv.tv_sec)+1/1000000.0*static_cast<double>(start_tv.tv_usec))) * 1000.0;

  lapse_time2 = ((static_cast<double>(stop2_tv.tv_sec)+1/1000000.0*static_cast<double>(stop2_tv.tv_usec)) - (static_cast<double>(stop1_tv.tv_sec)+1/1000000.0*static_cast<double>(stop1_tv.tv_usec))) * 1000.0;

  lapse_time3 = ((static_cast<double>(stop3_tv.tv_sec)+1/1000000.0*static_cast<double>(stop3_tv.tv_usec)) - (static_cast<double>(stop2_tv.tv_sec)+1/1000000.0*static_cast<double>(stop2_tv.tv_usec))) * 1000.0;

  lapse_time4 = ((static_cast<double>(stop3_tv.tv_sec)+1/1000000.0*static_cast<double>(stop3_tv.tv_usec)) - (static_cast<double>(start_tv.tv_sec)+1/1000000.0*static_cast<double>(start_tv.tv_usec))) * 1000.0;

  MPI_Barrier(MPI_COMM_WORLD);
  file << "TIME PACK: " << lapse_time1 << std::endl;
  file << "TIME EXCH: " << lapse_time2 << std::endl;
  file << "TIME UNPK: " << lapse_time3 << std::endl;
  file << "TIME ALL : " << lapse_time1+lapse_time2+lapse_time3 << std::endl;
  file << "TIME TOT : " << lapse_time4 << std::endl;

  status = cudaMemcpy( a.ptr, gpu_a,
                       (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)
                       *sizeof(triple_t<USE_DOUBLE>::data_type), 
                       cudaMemcpyDeviceToHost );
  if( !checkCudaStatus( status ) ) return;

  status = cudaMemcpy( b.ptr, gpu_b,
                       (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)
                       *sizeof(triple_t<USE_DOUBLE>::data_type), 
                       cudaMemcpyDeviceToHost );
  if( !checkCudaStatus( status ) ) return;

  status = cudaMemcpy( c.ptr, gpu_c,
                       (DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)
                       *sizeof(triple_t<USE_DOUBLE>::data_type), 
                       cudaMemcpyDeviceToHost );
  if( !checkCudaStatus( status ) ) return;

  status = cudaFree( gpu_a );
  if( !checkCudaStatus( status ) ) return;
  status = cudaFree( gpu_b );
  if( !checkCudaStatus( status ) ) return;
  status = cudaFree( gpu_c );
  if( !checkCudaStatus( status ) ) return;

  file << "\n********************************************************************************\n";

  file << "A \n";
  printbuff(file,a, DIM1+H1m+H1p, DIM2+H2m+H2p, DIM3+H3m+H3p);
  file << "B \n";
  printbuff(file,b, DIM1+H1m+H1p, DIM2+H2m+H2p, DIM3+H3m+H3p);
  file << "C \n";
  printbuff(file,c, DIM1+H1m+H1p, DIM2+H2m+H2p, DIM3+H3m+H3p);
  file.flush();
  file.flush();
  int passed = true;


  /* Checking the data arrived correctly in the whole region
   */
  for (int ii=0; ii<DIM1+H1m+H1p; ++ii)
    for (int jj=0; jj<DIM2+H2m+H2p; ++jj)
      for (int kk=0; kk<DIM3+H3m+H3p; ++kk) {

        triple_t<USE_DOUBLE> ta;
        triple_t<USE_DOUBLE> tb;
        triple_t<USE_DOUBLE> tc;
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
            tax=triple_t<USE_DOUBLE>().x();
            tbx=triple_t<USE_DOUBLE>().x();
            tcx=triple_t<USE_DOUBLE>().x();
          }
        }

        if (!per1) {
          if ( ((coords[1]==0) && (jj<H2m)) || 
               ((coords[1] == dims[1]-1) && (jj >= DIM2+H2m)) ) {
            tay=triple_t<USE_DOUBLE>().y();
            tby=triple_t<USE_DOUBLE>().y();
            tcy=triple_t<USE_DOUBLE>().y();
          }
        }

        if (!per2) {
          if ( ((coords[2]==0) && (kk<H3m)) || 
               ((coords[2] == dims[2]-1) && (kk >= DIM3+H3m)) ) {
            taz=triple_t<USE_DOUBLE>().z();
            tbz=triple_t<USE_DOUBLE>().z();
            tcz=triple_t<USE_DOUBLE>().z();
          }
        }

        ta = triple_t<USE_DOUBLE>(tax, tay, taz).floor();
        tb = triple_t<USE_DOUBLE>(tbx, tby, tbz).floor();
        tc = triple_t<USE_DOUBLE>(tcx, tcy, tcz).floor();

        if (a(ii,jj,kk) != ta) {
          passed = false;
          file << ii << ", " << jj << ", " << kk << " values found != expct: " 
               << "a " << a(ii,jj,kk) << " != " 
               << ta
               << "\n";
        }

        if (b(ii,jj,kk) != tb) {
          passed = false;
          file << ii << ", " << jj << ", " << kk << " values found != expct: " 
               << "b " << b(ii,jj,kk) << " != " 
               << tb
               << "\n";
        }

        if (c(ii,jj,kk) != tc) {
          passed = false;
          file << ii << ", " << jj << ", " << kk << " values found != expct: " 
               << "c " << c(ii,jj,kk) << " != " 
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
  int H1m  =atoi(argv[4]);
  int H1p  =atoi(argv[5]);
  int H2m  =atoi(argv[6]);
  int H2p  =atoi(argv[7]);
  int H3m  =atoi(argv[8]);
  int H3p  =atoi(argv[9]);


  /* This example will exchange 3 data arrays at the same time with
     different values.
   */
  triple_t<USE_DOUBLE> *_a = new triple_t<USE_DOUBLE>[(DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)];
  triple_t<USE_DOUBLE> *_b = new triple_t<USE_DOUBLE>[(DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)];
  triple_t<USE_DOUBLE> *_c = new triple_t<USE_DOUBLE>[(DIM1+H1m+H1p)*(DIM2+H2m+H2p)*(DIM3+H3m+H3p)];


  file << "Permutation 0,1,2\n";

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
