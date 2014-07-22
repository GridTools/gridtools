#include <mpi.h>
#include <communication/GCL.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <communication/low-level/proc_grids_2D.h>
#include <common/boollist.h>

#define OUT(I,J)  << "( " << I << ", " << J << ") <" << pg.proc<I,J>() << "> - "

std::ostream *filep;

int _main(int pid, int nprocs) {

  gridtools::_2D_process_grid_t<gridtools::boollist<2> > pg(gridtools::boollist<2>(true, true), nprocs,pid);

  (*filep) << "@" << gridtools::PID << "@ --- PROC GRID " 
    OUT(-1,-1)
    OUT(0,-1)
    OUT(1,0-1)
    OUT(-1,0)
    OUT(0,0)
    OUT(1,0)
    OUT(-1,1)
    OUT(0,1)
    OUT(1,1)
            << "\n";

  return 0;
}

int MPImain(MPI_Comm &comm) {

  gridtools::MPI_2D_process_grid_t<gridtools::boollist<2> > pg(gridtools::boollist<2>(true,true), comm);

  (*filep) << "@" << gridtools::PID << "@ MPI PROC GRID " 
    OUT(-1,-1)
    OUT(0,-1)
    OUT(1,0-1)
    OUT(-1,0)
    OUT(0,0)
    OUT(1,0)
    OUT(-1,1)
    OUT(0,1)
    OUT(1,1)
            << "\n";

  return 0;
}

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);
  gridtools::GCL_Init(argc, argv);

  int pid;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  std::stringstream ss;
  ss << pid;

  std::string filename = "out" + ss.str() + ".txt";

  std::cout << filename << std::endl;
  std::ofstream file(filename.c_str());

  filep = &file;

  _main(pid, nprocs);

  MPI_Comm CartComm;
  int dims[2] = {0,0};
  MPI_Dims_create(nprocs, 2, dims);
  int period[2] = {1, 1};

  file << "@" << gridtools::PID << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << "\n";
 
  MPI_Cart_create(gridtools::GCL_WORLD, 2, dims, period, false, &CartComm);

  MPImain(CartComm);

  return 0;
}
