#include <mpi.h>
#include <iostream>
#include <proc_grids_3D.hpp>
#include <Halo_Exchange_3D.hpp>
#include <stdio.h>
#include <common/boollist.hpp>

struct T3 {}; // MPI CYCLIC
struct T4 {}; // MPI not CYCLIC

template <typename T>
struct pgrid;

template <>
struct pgrid<T3> {

  typedef gridtools::MPI_3D_process_grid_t<3> grid_type;

  static grid_type instantiate(MPI_Comm comm) {
    int pid;
    MPI_Comm_rank(gridtools::GCL_WORLD, &pid);
    int nprocs;
    MPI_Comm_size(gridtools::GCL_WORLD, &nprocs);
    MPI_Comm CartComm;
    int dims[3] = {0,0,0};
    MPI_Dims_create(nprocs, 3, dims);
    int period[3] = {1, 1, 1};

    std::cout << "@" << gridtools::PID << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1]  << " - " << dims[2] << "\n";

    MPI_Cart_create(gridtools::GCL_WORLD, 3, dims, period, false, &CartComm);

    return grid_type(gridtools::boollist<3>(true, true, true), CartComm);
  }

};

template <>
struct pgrid<T4> {

  typedef gridtools::MPI_3D_process_grid_t<3> grid_type;

  static grid_type instantiate(MPI_Comm comm) {
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm CartComm;
    int dims[3] = {0,0,0};
    MPI_Dims_create(nprocs, 3, dims);
    int period[3] = {0, 0, 0};

    std::cout << "@" << gridtools::PID << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << " - " << dims[2] << "\n";

    MPI_Cart_create(gridtools::GCL_WORLD, 3, dims, period, false, &CartComm);

    return grid_type(gridtools::boollist<3>(false,false,false), CartComm);
  }

};

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);
  gridtools::GCL_Init(argc, argv);

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

  gridtools::Halo_Exchange_3D<test_type::grid_type> he(pg);

  std::cout << "@" << gridtools::PID << "@ SEND "
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
  std::cout << "@" << gridtools::PID << "@ RECV "
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



  iminus = gridtools::PID;
  iplus = gridtools::PID;
  jminus = gridtools::PID;
  jplus = gridtools::PID;
  kminus = gridtools::PID;
  kplus = gridtools::PID;
  iminusjminus = gridtools::PID;
  iplusjminus = gridtools::PID;
  iminusjplus = gridtools::PID;
  iplusjplus = gridtools::PID;
  iminuskminus = gridtools::PID;
  ipluskminus = gridtools::PID;
  iminuskplus = gridtools::PID;
  ipluskplus = gridtools::PID;
  jminuskminus = gridtools::PID;
  jpluskminus = gridtools::PID;
  jminuskplus = gridtools::PID;
  jpluskplus = gridtools::PID;
  iminusjminuskminus = gridtools::PID;
  iplusjminuskminus = gridtools::PID;
  iminusjpluskminus = gridtools::PID;
  iplusjpluskminus = gridtools::PID;
  iminusjminuskplus = gridtools::PID;
  iplusjminuskplus = gridtools::PID;
  iminusjpluskplus = gridtools::PID;
  iplusjpluskplus = gridtools::PID;


  printf("@%3d@ ----------------\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ ----------------\n\n",
         gridtools::PID,
         gridtools::PID, iminusjminuskminus, jminuskminus, iplusjminuskminus,
         gridtools::PID, iminuskminus, kminus, ipluskminus,
         gridtools::PID, iminusjpluskminus, jpluskminus, iplusjpluskminus,
         gridtools::PID);

  printf("@%3d@ ----------------\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ ----------------\n\n",
         gridtools::PID,
         gridtools::PID, iminusjminus, jminus, iplusjminus,
         gridtools::PID, iminus, gridtools::PID, iplus,
         gridtools::PID, iminusjplus, jplus, iplusjplus,
         gridtools::PID);

  printf("@%3d@ ----------------\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ ----------------\n\n",
         gridtools::PID,
         gridtools::PID, iminusjminuskplus, jminuskplus, iplusjminuskplus,
         gridtools::PID, iminuskplus, kplus, ipluskplus,
         gridtools::PID, iminusjpluskplus, jpluskplus, iplusjpluskplus,
         gridtools::PID);

  he.exchange();

  printf("@%3d@ ----------------\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ ----------------\n\n",
         gridtools::PID,
         gridtools::PID, iminusjminuskminus_r, jminuskminus_r, iplusjminuskminus_r,
         gridtools::PID, iminuskminus_r, kminus_r, ipluskminus_r,
         gridtools::PID, iminusjpluskminus_r, jpluskminus_r, iplusjpluskminus_r,
         gridtools::PID);

  printf("@%3d@ ----------------\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ ----------------\n\n",
         gridtools::PID,
         gridtools::PID, iminusjminus_r, jminus_r, iplusjminus_r,
         gridtools::PID, iminus_r, gridtools::PID, iplus_r,
         gridtools::PID, iminusjplus_r, jplus_r, iplusjplus_r,
         gridtools::PID);

  printf("@%3d@ ----------------\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ |%3d |%3d |%3d |\n@%3d@ ----------------\n\n",
         gridtools::PID,
         gridtools::PID, iminusjminuskplus_r, jminuskplus_r, iplusjminuskplus_r,
         gridtools::PID, iminuskplus_r, kplus_r, ipluskplus_r,
         gridtools::PID, iminusjpluskplus_r, jpluskplus_r, iplusjpluskplus_r,
         gridtools::PID);

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
