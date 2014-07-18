
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
#include <GCL.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <proc_grids_3D.h>
#include <utils/boollist.h>

#define OUT(I,J,K)  << "( " << I << ", " << J << ", " << K << ") " << pg.proc<I,J,K>() << " - "

std::ostream *filep;

int _main(int pid, int nprocs) {

  GCL::_3D_process_grid_t<GCL::gcl_utils::boollist<3> > pg(GCL::gcl_utils::boollist<3>(true,true,true), nprocs,pid);

  int R_,C_,S_;
  pg.dims(R_,C_,S_);
  (*filep) << "@" << GCL::PID << "@ GRID SIZE " << R_ << " - " << C_ << " - " << S_ << "\n";  

  (*filep) << "@" << GCL::PID << "@ PROC GRID " 
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

int MPImain(MPI_Comm &comm) {

  GCL::MPI_3D_process_grid_t<GCL::gcl_utils::boollist<3> > pg(GCL::gcl_utils::boollist<3>(true, true, true), comm);

  int R_,C_,S_;
  pg.dims(R_,C_,S_);
  (*filep) << "@" << GCL::PID << "@ GRID SIZE " << R_ << " - " << C_ << " - " << S_ << "\n";  

  (*filep) << "@" << GCL::PID << "@ PROC GRID " 
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
  GCL::GCL_Init(argc, argv);

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
  int dims[3] = {0,0,0};
  MPI_Dims_create(nprocs, 3, dims);
  int period[3] = {1, 1, 1};

  std::cout << "@" << GCL::PID << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << " - " << dims[2] << "\n";
 
  MPI_Cart_create(GCL::GCL_WORLD, 3, dims, period, false, &CartComm);

  MPImain(CartComm);

  MPI_Finalize();

  return 0;
}
