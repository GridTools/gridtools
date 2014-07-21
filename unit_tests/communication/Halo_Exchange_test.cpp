
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
#include <proc_grids_2D.h>
#include <Halo_Exchange_2D.h>
#include <stdio.h>
#include <common/boollist.h>

struct T1 {}; // GCL CYCLIC
struct T2 {}; // GCL not CYCLIC
struct T3 {}; // MPI CYCLIC
struct T4 {}; // MPI not CYCLIC

template <typename T>
struct pgrid;

template <>
struct pgrid<T1> {

  typedef gridtools::_2D_process_grid_t<gridtools::gcl_utils::boollist<2> > grid_type;

  static grid_type instantiate(MPI_Comm comm) {
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    return grid_type(gridtools::gcl_utils::boollist<2>(true,true), nprocs, pid);
  }

};

template <>
struct pgrid<T2> {

  typedef gridtools::_2D_process_grid_t<gridtools::gcl_utils::boollist<2> > grid_type;

  static grid_type instantiate(MPI_Comm comm) {
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    return grid_type(gridtools::gcl_utils::boollist<2>(false,false) ,nprocs, pid);
  }

};

template <>
struct pgrid<T3> {

  typedef gridtools::MPI_2D_process_grid_t<gridtools::gcl_utils::boollist<2> > grid_type;

  static grid_type instantiate(MPI_Comm comm) {
    int pid;
    MPI_Comm_rank(gridtools::GCL_WORLD, &pid);
    int nprocs;
    MPI_Comm_size(gridtools::GCL_WORLD, &nprocs);
    MPI_Comm CartComm;
    int dims[2] = {0,0};
    MPI_Dims_create(nprocs, 2, dims);
    int period[2] = {1, 1};

    std::cout << "@" << gridtools::PID << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << "\n";
 
    MPI_Cart_create(gridtools::GCL_WORLD, 2, dims, period, false, &CartComm);

    return grid_type(gridtools::gcl_utils::boollist<2>(true,true), CartComm);
  }

};

template <>
struct pgrid<T4> {

  typedef gridtools::MPI_2D_process_grid_t<gridtools::gcl_utils::boollist<2> > grid_type;

  static grid_type instantiate(MPI_Comm comm) {
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm CartComm;
    int dims[2] = {0,0};
    MPI_Dims_create(nprocs, 2, dims);
    int period[2] = {0, 0};

    std::cout << "@" << gridtools::PID << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << "\n";
 
    MPI_Cart_create(gridtools::GCL_WORLD, 2, dims, period, false, &CartComm);

    return grid_type(gridtools::gcl_utils::boollist<2>(false, false), CartComm);
  }

};

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);
  gridtools::GCL_Init(argc, argv);

  int iminus;
  int iplus;
  int jminus;
  int jplus;
  int iminusjminus;
  int iplusjminus;
  int iminusjplus;
  int iplusjplus;

  int iminus_r=-1;
  int iplus_r=-1;
  int jminus_r=-1;
  int jplus_r=-1;
  int iminusjminus_r=-1;
  int iplusjminus_r=-1;
  int iminusjplus_r=-1;
  int iplusjplus_r=-1;

  typedef pgrid<T3> test_type;

  test_type::grid_type pg = test_type::instantiate(MPI_COMM_WORLD);

  gridtools::Halo_Exchange_2D<test_type::grid_type> he(pg);

  std::cout << "@" << gridtools::PID << "@ SEND " 
            << &iminus << " - " 
            << &iplus << " - " 
            << &jminus << " - " 
            << &jplus << " - " 
            << &iminusjminus << " - " 
            << &iplusjminus << " - " 
            << &iminusjplus << " - " 
            << &iplusjplus << std::endl; 
  std::cout << "@" << gridtools::PID << "@ RECV " 
            << &iminus_r << " - " 
            << &iplus_r << " - " 
            << &jminus_r << " - " 
            << &jplus_r << " - " 
            << &iminusjminus_r << " - " 
            << &iplusjminus_r << " - " 
            << &iminusjplus_r << " - " 
            << &iplusjplus_r << std::endl;

  he.register_send_to_buffer<-1,-1>(&iminusjminus, sizeof(int));
  he.register_send_to_buffer<-1, 1>(&iminusjplus, sizeof(int));
  he.register_send_to_buffer< 1,-1>(&iplusjminus, sizeof(int));
  he.register_send_to_buffer< 1, 1>(&iplusjplus, sizeof(int));
  he.register_send_to_buffer<-1, 0>(&iminus, sizeof(int));
  he.register_send_to_buffer< 1, 0>(&iplus, sizeof(int));
  he.register_send_to_buffer< 0,-1>(&jminus, sizeof(int));
  he.register_send_to_buffer< 0, 1>(&jplus, sizeof(int));

  he.register_receive_from_buffer<-1,-1>(&iminusjminus_r, sizeof(int));
  he.register_receive_from_buffer<-1, 1>(&iminusjplus_r, sizeof(int));
  he.register_receive_from_buffer< 1,-1>(&iplusjminus_r, sizeof(int));
  he.register_receive_from_buffer< 1, 1>(&iplusjplus_r, sizeof(int));
  he.register_receive_from_buffer<-1, 0>(&iminus_r, sizeof(int));
  he.register_receive_from_buffer< 1, 0>(&iplus_r, sizeof(int));
  he.register_receive_from_buffer< 0,-1>(&jminus_r, sizeof(int));
  he.register_receive_from_buffer< 0, 1>(&jplus_r, sizeof(int));

  iminus = gridtools::PID;
  iplus = gridtools::PID;
  jminus = gridtools::PID;
  jplus = gridtools::PID;
  iminusjminus = gridtools::PID;
  iplusjminus = gridtools::PID;
  iminusjplus = gridtools::PID;
  iplusjplus = gridtools::PID;

  he.exchange();

  printf("@%3d@ ----------------\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ ----------------\n\n",
         gridtools::PID,
         gridtools::PID, iminusjminus_r, iminus_r, iminusjplus_r,
         gridtools::PID, jminus_r, gridtools::PID, jplus_r,
         gridtools::PID, iplusjminus_r, iplus_r, iplusjplus_r,
         gridtools::PID);

  int res = 1;

  res &= (pg.proc(-1,-1) == iminusjminus_r);
  res &= (pg.proc(-1, 0) == iminus_r);
  res &= (pg.proc(-1, 1) == iminusjplus_r);
  res &= (pg.proc( 0,-1) == jminus_r);
  res &= (pg.proc( 0, 1) == jplus_r);
  res &= (pg.proc( 1,-1) == iplusjminus_r);
  res &= (pg.proc( 1, 0) == iplus_r);
  res &= (pg.proc( 1, 1) == iplusjplus_r);

  int final;
  MPI_Reduce(&res, &final, 1, MPI_INT, MPI_LAND, 0, gridtools::GCL_WORLD);

  if (gridtools::PID==0) {
    if (!final) {
      std::cout << "@" << gridtools::PID << "@ FAILED!\n";
    } else
      std::cout << "@" << gridtools::PID << "@ PASSED!\n";
  }

  MPI_Barrier(gridtools::GCL_WORLD);
  MPI_Finalize();
  return !final;
}
