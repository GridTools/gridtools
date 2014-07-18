
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
#include <proc_grids_3D.h>
#include <Halo_Exchange_3D.h>
#include <stdio.h>
#include <utils/boollist.h>

struct T1 {}; // GCL CYCLIC
struct T2 {}; // GCL not CYCLIC
struct T3 {}; // MPI CYCLIC
struct T4 {}; // MPI not CYCLIC

template <typename T>
struct pgrid;

template <>
struct pgrid<T1> {

  typedef GCL::_3D_process_grid_t<GCL::gcl_utils::boollist<3> > grid_type;

  static grid_type instantiate(MPI_Comm comm) {
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    return grid_type(GCL::gcl_utils::boollist<3>(true, true, true), nprocs, pid);
  }

};

template <>
struct pgrid<T2> {

  typedef GCL::_3D_process_grid_t<GCL::gcl_utils::boollist<3> > grid_type;

  static grid_type instantiate(MPI_Comm comm) {
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    return grid_type(GCL::gcl_utils::boollist<3>(false,false,false), nprocs, pid);
  }

};

template <>
struct pgrid<T3> {

  typedef GCL::MPI_3D_process_grid_t<GCL::gcl_utils::boollist<3> > grid_type;

  static grid_type instantiate(MPI_Comm comm) {
    int pid;
    MPI_Comm_rank(GCL::GCL_WORLD, &pid);
    int nprocs;
    MPI_Comm_size(GCL::GCL_WORLD, &nprocs);
    MPI_Comm CartComm;
    int dims[3] = {0,0,0};
    MPI_Dims_create(nprocs, 3, dims);
    int period[3] = {1, 1, 1};

    std::cout << "@" << GCL::PID << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1]  << " - " << dims[2] << "\n";
 
    MPI_Cart_create(GCL::GCL_WORLD, 3, dims, period, false, &CartComm);

    return grid_type(GCL::gcl_utils::boollist<3>(true, true, true), CartComm);
  }

};

template <>
struct pgrid<T4> {

  typedef GCL::MPI_3D_process_grid_t<GCL::gcl_utils::boollist<3> > grid_type;

  static grid_type instantiate(MPI_Comm comm) {
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm CartComm;
    int dims[3] = {0,0,0};
    MPI_Dims_create(nprocs, 3, dims);
    int period[3] = {0, 0, 0};

    std::cout << "@" << GCL::PID << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << " - " << dims[2] << "\n";
 
    MPI_Cart_create(GCL::GCL_WORLD, 3, dims, period, false, &CartComm);

    return grid_type(GCL::gcl_utils::boollist<3>(false,false,false), CartComm);
  }

};

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);
  GCL::GCL_Init(argc, argv);

  //6
  int iminus;
  int iplus;
  int jminus;
  int jplus;
  int kminus;
  int kplus;

  //12
  int iminusjminus;
  int iplusjminus;
  int iminusjplus;
  int iplusjplus;
  int iminuskminus;
  int ipluskminus;
  int iminuskplus;
  int ipluskplus;
  int jminuskminus;
  int jpluskminus;
  int jminuskplus;
  int jpluskplus;

  //8
  int iminusjminuskminus;
  int iplusjminuskminus;
  int iminusjpluskminus;
  int iplusjpluskminus;
  int iminusjminuskplus;
  int iplusjminuskplus;
  int iminusjpluskplus;
  int iplusjpluskplus;

  int iminus_r=-1;
  int iplus_r=-1;
  int jminus_r=-1;
  int jplus_r=-1;
  int kminus_r=-1;
  int kplus_r=-1;

  int iminusjminus_r=-1;
  int iplusjminus_r=-1;
  int iminusjplus_r=-1;
  int iplusjplus_r=-1;
  int iminuskminus_r=-1;
  int ipluskminus_r=-1;
  int iminuskplus_r=-1;
  int ipluskplus_r=-1;
  int jminuskminus_r=-1;
  int jpluskminus_r=-1;
  int jminuskplus_r=-1;
  int jpluskplus_r=-1;

  int iminusjminuskminus_r=-1;
  int iplusjminuskminus_r=-1;
  int iminusjpluskminus_r=-1;
  int iplusjpluskminus_r=-1;
  int iminusjminuskplus_r=-1;
  int iplusjminuskplus_r=-1;
  int iminusjpluskplus_r=-1;
  int iplusjpluskplus_r=-1;

  typedef pgrid<T3> test_type;

  test_type::grid_type pg = test_type::instantiate(MPI_COMM_WORLD);

  GCL::Halo_Exchange_3D<test_type::grid_type> he(pg);

  std::cout << "@" << GCL::PID << "@ SEND " 
            << &iminus << " - " 
            << &iplus << " - " 
            << &jminus << " - " 
            << &jplus << " - " 
            << &kminus << " - " 
            << &kplus << " - " 
            << &iminusjminus << " - " 
            << &iplusjminus << " - " 
            << &iminusjplus << " - " 
            << &iplusjplus << " - " 
            << &iminuskminus << " - " 
            << &ipluskminus << " - " 
            << &iminuskplus << " - " 
            << &ipluskplus << " - "
            << &jminuskminus << " - " 
            << &jpluskminus << " - " 
            << &jminuskplus << " - " 
            << &jpluskplus << " - "
            << &iminusjminuskminus << " - " 
            << &iplusjminuskminus << " - " 
            << &iminusjpluskminus << " - " 
            << &iplusjpluskminus << " - " 
            << &iminusjminuskplus << " - " 
            << &iplusjminuskplus << " - " 
            << &iminusjpluskplus << " - " 
            << &iplusjpluskplus << std::endl; 
  std::cout << "@" << GCL::PID << "@ RECV " 
            << &iminus_r << " - " 
            << &iplus_r << " - " 
            << &jminus_r << " - " 
            << &jplus_r << " - " 
            << &kminus_r << " - " 
            << &kplus_r << " - " 
            << &iminusjminus_r << " - " 
            << &iplusjminus_r << " - " 
            << &iminusjplus_r << " - " 
            << &iplusjplus_r << " - " 
            << &iminuskminus_r << " - " 
            << &ipluskminus_r << " - " 
            << &iminuskplus_r << " - " 
            << &ipluskplus_r << " - "
            << &jminuskminus_r << " - " 
            << &jpluskminus_r << " - " 
            << &jminuskplus_r << " - " 
            << &jpluskplus_r << " - "
            << &iminusjminuskminus_r << " - " 
            << &iplusjminuskminus_r << " - " 
            << &iminusjpluskminus_r << " - " 
            << &iplusjpluskminus_r << " - " 
            << &iminusjminuskplus_r << " - " 
            << &iplusjminuskplus_r << " - " 
            << &iminusjpluskplus_r << " - " 
            << &iplusjpluskplus_r << std::endl; 

  he.register_send_to_buffer<-1,-1,-1>(&iminusjminuskminus, sizeof(int));
  he.register_send_to_buffer<-1, 1,-1>(&iminusjpluskminus, sizeof(int));
  he.register_send_to_buffer< 1,-1,-1>(&iplusjminuskminus, sizeof(int));
  he.register_send_to_buffer< 1, 1,-1>(&iplusjpluskminus, sizeof(int));
  he.register_send_to_buffer<-1,-1, 1>(&iminusjminuskplus, sizeof(int));
  he.register_send_to_buffer<-1, 1, 1>(&iminusjpluskplus, sizeof(int));
  he.register_send_to_buffer< 1,-1, 1>(&iplusjminuskplus, sizeof(int));
  he.register_send_to_buffer< 1, 1, 1>(&iplusjpluskplus, sizeof(int));

  he.register_send_to_buffer<-1,-1, 0>(&iminusjminus, sizeof(int));
  he.register_send_to_buffer<-1, 1, 0>(&iminusjplus, sizeof(int));
  he.register_send_to_buffer< 1,-1, 0>(&iplusjminus, sizeof(int));
  he.register_send_to_buffer< 1, 1, 0>(&iplusjplus, sizeof(int));

  he.register_send_to_buffer<-1, 0,-1>(&iminuskminus, sizeof(int));
  he.register_send_to_buffer<-1, 0, 1>(&iminuskplus, sizeof(int));
  he.register_send_to_buffer< 1, 0,-1>(&ipluskminus, sizeof(int));
  he.register_send_to_buffer< 1, 0, 1>(&ipluskplus, sizeof(int));

  he.register_send_to_buffer< 0,-1,-1>(&jminuskminus, sizeof(int));
  he.register_send_to_buffer< 0,-1, 1>(&jminuskplus, sizeof(int));
  he.register_send_to_buffer< 0, 1,-1>(&jpluskminus, sizeof(int));
  he.register_send_to_buffer< 0, 1, 1>(&jpluskplus, sizeof(int));

  he.register_send_to_buffer<-1, 0, 0>(&iminus, sizeof(int));
  he.register_send_to_buffer< 1, 0, 0>(&iplus, sizeof(int));
  he.register_send_to_buffer< 0,-1, 0>(&jminus, sizeof(int));
  he.register_send_to_buffer< 0, 1, 0>(&jplus, sizeof(int));
  he.register_send_to_buffer< 0, 0,-1>(&kminus, sizeof(int));
  he.register_send_to_buffer< 0, 0, 1>(&kplus, sizeof(int));


  he.register_receive_from_buffer<-1,-1,-1>(&iminusjminuskminus_r, sizeof(int));
  he.register_receive_from_buffer<-1, 1,-1>(&iminusjpluskminus_r, sizeof(int));
  he.register_receive_from_buffer< 1,-1,-1>(&iplusjminuskminus_r, sizeof(int));
  he.register_receive_from_buffer< 1, 1,-1>(&iplusjpluskminus_r, sizeof(int));
  he.register_receive_from_buffer<-1,-1, 1>(&iminusjminuskplus_r, sizeof(int));
  he.register_receive_from_buffer<-1, 1, 1>(&iminusjpluskplus_r, sizeof(int));
  he.register_receive_from_buffer< 1,-1, 1>(&iplusjminuskplus_r, sizeof(int));
  he.register_receive_from_buffer< 1, 1, 1>(&iplusjpluskplus_r, sizeof(int));

  he.register_receive_from_buffer<-1,-1, 0>(&iminusjminus_r, sizeof(int));
  he.register_receive_from_buffer<-1, 1, 0>(&iminusjplus_r, sizeof(int));
  he.register_receive_from_buffer< 1,-1, 0>(&iplusjminus_r, sizeof(int));
  he.register_receive_from_buffer< 1, 1, 0>(&iplusjplus_r, sizeof(int));

  he.register_receive_from_buffer<-1, 0,-1>(&iminuskminus_r, sizeof(int));
  he.register_receive_from_buffer<-1, 0, 1>(&iminuskplus_r, sizeof(int));
  he.register_receive_from_buffer< 1, 0,-1>(&ipluskminus_r, sizeof(int));
  he.register_receive_from_buffer< 1, 0, 1>(&ipluskplus_r, sizeof(int));

  he.register_receive_from_buffer< 0,-1,-1>(&jminuskminus_r, sizeof(int));
  he.register_receive_from_buffer< 0,-1, 1>(&jminuskplus_r, sizeof(int));
  he.register_receive_from_buffer< 0, 1,-1>(&jpluskminus_r, sizeof(int));
  he.register_receive_from_buffer< 0, 1, 1>(&jpluskplus_r, sizeof(int));

  he.register_receive_from_buffer<-1, 0, 0>(&iminus_r, sizeof(int));
  he.register_receive_from_buffer< 1, 0, 0>(&iplus_r, sizeof(int));
  he.register_receive_from_buffer< 0,-1, 0>(&jminus_r, sizeof(int));
  he.register_receive_from_buffer< 0, 1, 0>(&jplus_r, sizeof(int));
  he.register_receive_from_buffer< 0, 0,-1>(&kminus_r, sizeof(int));
  he.register_receive_from_buffer< 0, 0, 1>(&kplus_r, sizeof(int));



  iminus = GCL::PID;
  iplus = GCL::PID;
  jminus = GCL::PID;
  jplus = GCL::PID;
  kminus = GCL::PID;
  kplus = GCL::PID;
  iminusjminus = GCL::PID;
  iplusjminus = GCL::PID;
  iminusjplus = GCL::PID;
  iplusjplus = GCL::PID;
  iminuskminus = GCL::PID;
  ipluskminus = GCL::PID;
  iminuskplus = GCL::PID;
  ipluskplus = GCL::PID;
  jminuskminus = GCL::PID;
  jpluskminus = GCL::PID;
  jminuskplus = GCL::PID;
  jpluskplus = GCL::PID;
  iminusjminuskminus = GCL::PID;
  iplusjminuskminus = GCL::PID;
  iminusjpluskminus = GCL::PID;
  iplusjpluskminus = GCL::PID;
  iminusjminuskplus = GCL::PID;
  iplusjminuskplus = GCL::PID;
  iminusjpluskplus = GCL::PID;
  iplusjpluskplus = GCL::PID;


  printf("@%3d@ ----------------\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ ----------------\n\n",
         GCL::PID,
         GCL::PID, iminusjminuskminus, jminuskminus, iplusjminuskminus,
         GCL::PID, iminuskminus, kminus, ipluskminus,
         GCL::PID, iminusjpluskminus, jpluskminus, iplusjpluskminus,
         GCL::PID);

  printf("@%3d@ ----------------\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ ----------------\n\n",
         GCL::PID,
         GCL::PID, iminusjminus, jminus, iplusjminus,
         GCL::PID, iminus, GCL::PID, iplus,
         GCL::PID, iminusjplus, jplus, iplusjplus,
         GCL::PID);

  printf("@%3d@ ----------------\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ ----------------\n\n",
         GCL::PID,
         GCL::PID, iminusjminuskplus, jminuskplus, iplusjminuskplus,
         GCL::PID, iminuskplus, kplus, ipluskplus,
         GCL::PID, iminusjpluskplus, jpluskplus, iplusjpluskplus,
         GCL::PID);

  he.exchange();

  printf("@%3d@ ----------------\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ ----------------\n\n",
         GCL::PID,
         GCL::PID, iminusjminuskminus_r, jminuskminus_r, iplusjminuskminus_r,
         GCL::PID, iminuskminus_r, kminus_r, ipluskminus_r,
         GCL::PID, iminusjpluskminus_r, jpluskminus_r, iplusjpluskminus_r,
         GCL::PID);

  printf("@%3d@ ----------------\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ ----------------\n\n",
         GCL::PID,
         GCL::PID, iminusjminus_r, jminus_r, iplusjminus_r,
         GCL::PID, iminus_r, GCL::PID, iplus_r,
         GCL::PID, iminusjplus_r, jplus_r, iplusjplus_r,
         GCL::PID);

  printf("@%3d@ ----------------\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ ----------------\n\n",
         GCL::PID,
         GCL::PID, iminusjminuskplus_r, jminuskplus_r, iplusjminuskplus_r,
         GCL::PID, iminuskplus_r, kplus_r, ipluskplus_r,
         GCL::PID, iminusjpluskplus_r, jpluskplus_r, iplusjpluskplus_r,
         GCL::PID);

  int res = 1;

  res &= (pg.proc( 1, 0, 0) == iplus_r);
  res &= (pg.proc(-1, 0, 0) == iminus_r);
  res &= (pg.proc( 0, 1, 0) == jplus_r);
  res &= (pg.proc( 0,-1, 0) == jminus_r);
  res &= (pg.proc( 0, 0, 1) == kplus_r);
  res &= (pg.proc( 0, 0,-1) == kminus_r);

  res &= (pg.proc(-1, 1, 0) == iminusjplus_r);
  res &= (pg.proc( 1,-1, 0) == iplusjminus_r);
  res &= (pg.proc( 1, 1, 0) == iplusjplus_r);
  res &= (pg.proc(-1,-1, 0) == iminusjminus_r);

  res &= (pg.proc(-1, 0, 1) == iminuskplus_r);
  res &= (pg.proc( 1, 0,-1) == ipluskminus_r);
  res &= (pg.proc( 1, 0, 1) == ipluskplus_r);
  res &= (pg.proc(-1, 0,-1) == iminuskminus_r);

  res &= (pg.proc(0, -1, 1) == jminuskplus_r);
  res &= (pg.proc(0,  1,-1) == jpluskminus_r);
  res &= (pg.proc(0,  1, 1) == jpluskplus_r);
  res &= (pg.proc(0, -1,-1) == jminuskminus_r);

  res &= (pg.proc(-1, 1,-1) == iminusjpluskminus_r);
  res &= (pg.proc( 1,-1,-1) == iplusjminuskminus_r);
  res &= (pg.proc( 1, 1,-1) == iplusjpluskminus_r);
  res &= (pg.proc(-1,-1,-1) == iminusjminuskminus_r);

  res &= (pg.proc(-1, 1, 1) == iminusjpluskplus_r);
  res &= (pg.proc( 1,-1, 1) == iplusjminuskplus_r);
  res &= (pg.proc( 1, 1, 1) == iplusjpluskplus_r);
  res &= (pg.proc(-1,-1, 1) == iminusjminuskplus_r);

  int final;
  MPI_Reduce(&res, &final, 1, MPI_INT, MPI_LAND, 0, GCL::GCL_WORLD);

  if (GCL::PID==0) {
    if (!final) {
      std::cout << "@" << GCL::PID << "@ FAILED!\n";
    } else
      std::cout << "@" << GCL::PID << "@ PASSED!\n";
  }

  MPI_Barrier(GCL::GCL_WORLD);
  MPI_Finalize();
  return !final;
}
