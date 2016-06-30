#include <mpi.h>
#include <communication/GCL.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <communication/low-level/proc_grids_3D.hpp>
#include <common/boollist.hpp>

#define OUT(I,J,K)  << "( " << I << ", " << J << ", " << K << ") " << pg.proc<I,J,K>() << " - "

std::ostream *filep;


int MPImain(MPI_Comm &comm) {

  gridtools::MPI_3D_process_grid_t<3 > pg(gridtools::boollist<3>(true, true, true), comm);

  int R_,C_,S_;
  pg.dims(R_,C_,S_);
  (*filep) << "@" << gridtools::PID << "@ GRID SIZE " << R_ << " - " << C_ << " - " << S_ << "\n";

  (*filep) << "@" << gridtools::PID << "@ PROC GRID "
    OUT(-1,-1,-1)
    OUT(0,-1,-1)
    OUT(1,-1,-1)
    OUT(-1,0,-1)
    OUT(0,0,-1)
    OUT(1,0,-1)
    OUT(-1,1,-1)
    OUT(0,1,-1)
    OUT(1,1,-1)
    OUT(-1,-1,0)
    OUT(0,-1,0)
    OUT(1,-1,0)
    OUT(-1,0,0)
    OUT(0,0,0)
    OUT(1,0,0)
    OUT(-1,1,0)
    OUT(0,1,0)
    OUT(1,1,0)
    OUT(-1,-1,1)
    OUT(0,-1,1)
    OUT(1,-1,1)
    OUT(-1,0,1)
    OUT(0,0,1)
    OUT(1,0,1)
    OUT(-1,1,1)
    OUT(0,1,1)
    OUT(1,1,1)
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

  MPI_Comm CartComm;
  int dims[3] = {0,0,0};
  MPI_Dims_create(nprocs, 3, dims);
  int period[3] = {1, 1, 1};

  std::cout << "@" << gridtools::PID << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << " - " << dims[2] << "\n";

  MPI_Cart_create(gridtools::GCL_WORLD, 3, dims, period, false, &CartComm);

  MPImain(CartComm);

  MPI_Finalize();

  return 0;
}
