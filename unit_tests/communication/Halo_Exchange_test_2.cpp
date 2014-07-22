#include <mpi.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <proc_grids_2D.h>
#include <Halo_Exchange_2D.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <common/boollist.h>

struct T1 {}; // GCL CYCLIC
struct T2 {}; // GCL not CYCLIC
struct T3 {}; // MPI CYCLIC
struct T4 {}; // MPI not CYCLIC

template <typename T>
struct pgrid;


// THIS TEST DOES NOT WORK WITH CYCLIC GRIDS
template <>
struct pgrid<T1> {

  typedef gridtools::_2D_process_grid_t<gridtools::boollist<2> > grid_type;

  static grid_type instantiate(MPI_Comm comm) {
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    return grid_type(gridtools::boollist<2>(true,true), nprocs, pid);
  }

};

template <>
struct pgrid<T2> {

  typedef gridtools::_2D_process_grid_t<gridtools::boollist<2> > grid_type;

  static grid_type instantiate(MPI_Comm comm) {
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    return grid_type(gridtools::boollist<2>(false,false) ,nprocs, pid);
  }

};

template <>
struct pgrid<T3> {

  typedef gridtools::MPI_2D_process_grid_t<gridtools::boollist<2> > grid_type;

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

    return grid_type(gridtools::boollist<2>(true,true), CartComm);
  }

};

template <>
struct pgrid<T4> {

  typedef gridtools::MPI_2D_process_grid_t<gridtools::boollist<2> > grid_type;

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

    return grid_type(gridtools::boollist<2>(false, false), CartComm);
  }

};

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);
  gridtools::GCL_Init(argc, argv);

  std::stringstream ss;
  ss << gridtools::PID;

  std::string filename = "out" + ss.str() + ".txt";

  std::cout << filename << std::endl;
  std::ofstream file(filename.c_str());

  int k = 16;
  int i,j,N,M;
  
  typedef pgrid<T4> test_type;

  test_type::grid_type pg = test_type::instantiate(MPI_COMM_WORLD);

  gridtools::Halo_Exchange_2D<test_type::grid_type> he(pg);

  pg.coords(i,j);
  pg.dims(N,M);

  std::vector<int> iminus( k+i+j );
  std::vector<int> iplus( k+i+j );
  std::vector<int> jminus( k+i+j );
  std::vector<int> jplus( k+i+j );
  std::vector<int> iminusjminus( k+i+j );
  std::vector<int> iplusjminus( k+i+j );
  std::vector<int> iminusjplus( k+i+j );
  std::vector<int> iplusjplus( k+i+j );

  std::vector<int> iminus_r( k+i+j-1 );
  std::vector<int> iplus_r( k+i+j+1 );
  std::vector<int> jminus_r( k+i+j-1 );
  std::vector<int> jplus_r( k+i+j+1 );
  std::vector<int> iminusjminus_r( k+i+j-2 );
  std::vector<int> iplusjminus_r( k+i+j );
  std::vector<int> iminusjplus_r( k+i+j );
  std::vector<int> iplusjplus_r( k+i+j+2 );

  he.register_send_to_buffer<-1,-1>(&iminusjminus[0], (k+i+j)*sizeof(int));
  he.register_send_to_buffer<-1, 1>(&iminusjplus[0], (k+i+j)*sizeof(int));
  he.register_send_to_buffer< 1,-1>(&iplusjminus[0], (k+i+j)*sizeof(int));
  he.register_send_to_buffer< 1, 1>(&iplusjplus[0], (k+i+j)*sizeof(int));
  he.register_send_to_buffer<-1, 0>(&iminus[0], (k+i+j)*sizeof(int));
  he.register_send_to_buffer< 1, 0>(&iplus[0], (k+i+j)*sizeof(int));
  he.register_send_to_buffer< 0,-1>(&jminus[0], (k+i+j)*sizeof(int));
  he.register_send_to_buffer< 0, 1>(&jplus[0], (k+i+j)*sizeof(int));

  he.register_receive_from_buffer<-1,-1>(&iminusjminus_r[0], (k+i+j-2)*sizeof(int));
  he.register_receive_from_buffer<-1, 1>(&iminusjplus_r[0], (k+i+j)*sizeof(int));
  he.register_receive_from_buffer< 1,-1>(&iplusjminus_r[0], (k+i+j)*sizeof(int));
  he.register_receive_from_buffer< 1, 1>(&iplusjplus_r[0], (k+i+j+2)*sizeof(int));
  he.register_receive_from_buffer<-1, 0>(&iminus_r[0], (k+i+j-1)*sizeof(int));
  he.register_receive_from_buffer< 1, 0>(&iplus_r[0], (k+i+j+1)*sizeof(int));
  he.register_receive_from_buffer< 0,-1>(&jminus_r[0], (k+i+j-1)*sizeof(int));
  he.register_receive_from_buffer< 0, 1>(&jplus_r[0], (k+i+j+1)*sizeof(int));

  std::fill(&iminus[0], &iminus[k+i+j], gridtools::PID);
  std::fill(&iplus[0], &iplus[k+i+j], gridtools::PID);
  std::fill(&jminus[0], &jminus[k+i+j], gridtools::PID);
  std::fill(&jplus[0], &jplus[k+i+j], gridtools::PID);
  std::fill(&iminusjminus[0], &iminusjminus[k+i+j], gridtools::PID);
  std::fill(&iplusjplus[0], &iplusjplus[k+i+j], gridtools::PID);
  std::fill(&iplusjminus[0], &iplusjminus[k+i+j], gridtools::PID);
  std::fill(&iminusjplus[0], &iminusjplus[k+i+j], gridtools::PID);

  he.exchange();

  std::vector<int> res_iminus_r( k+i+j-1 );
  std::vector<int> res_iplus_r( k+i+j+1 );
  std::vector<int> res_jminus_r( k+i+j-1 );
  std::vector<int> res_jplus_r( k+i+j+1 );
  std::vector<int> res_iminusjminus_r( k+i+j-2 );
  std::vector<int> res_iplusjminus_r( k+i+j );
  std::vector<int> res_iminusjplus_r( k+i+j );
  std::vector<int> res_iplusjplus_r( k+i+j+2 );

  std::fill(&res_iminus_r[0], &res_iminus_r[k+i+j-1], pg.proc<-1,0>());
  std::fill(&res_iplus_r[0], &res_iplus_r[k+i+j+1], pg.proc<1,0>());
  std::fill(&res_jminus_r[0], &res_jminus_r[k+i+j-1], pg.proc<0,-1>());
  std::fill(&res_jplus_r[0], &res_jplus_r[k+i+j+1], pg.proc<0,1>());
  std::fill(&res_iminusjminus_r[0], &res_iminusjminus_r[k+i+j-2], pg.proc<-1,-1>());
  std::fill(&res_iplusjplus_r[0], &res_iplusjplus_r[k+i+j+2], pg.proc<1,1>());
  std::fill(&res_iplusjminus_r[0], &res_iplusjminus_r[k+i+j], pg.proc<1,-1>());
  std::fill(&res_iminusjplus_r[0], &res_iminusjplus_r[k+i+j], pg.proc<-1,1>());

  int res = 1;

  if (i>0) {
    res &= std::equal(&iminus_r[0], &iminus_r[k+i+j-1], &res_iminus_r[0]);
    file << gridtools::PID << " res = " <<  res << "\n";
  }
  if (i<N-1) {
    res &= std::equal(&iplus_r[0], &iplus_r[k+i+j+1], &res_iplus_r[0]);
    file << gridtools::PID << " res = " <<  res << "\n";
  }
  if (j>0) {
    res &= std::equal(&jminus_r[0], &jminus_r[k+i+j-1], &res_jminus_r[0]);
    file << gridtools::PID << " res = " <<  res << "\n";
  }
  if (j<M-1) {
    res &= std::equal(&jplus_r[0], &jplus_r[k+i+j+1], &res_jplus_r[0]);
    file << gridtools::PID << " res = " <<  res << "\n";
  }
  if (i>0 && j>0) {
    res &= std::equal(&iminusjminus_r[0], &iminusjminus_r[k+i+j-2], &res_iminusjminus_r[0]);
    file << gridtools::PID << " res = " <<  res << "\n";
  }
  if (i<N-1 && j>0) {
    res &= std::equal(&iplusjminus_r[0], &iplusjminus_r[k+i+j], &res_iplusjminus_r[0]);
    file << gridtools::PID << " res = " <<  res << "\n";
  }
  if (i>0 && j<M-1) {
    res &= std::equal(&iminusjplus_r[0], &iminusjplus_r[k+i+j], &res_iminusjplus_r[0]);
    file << gridtools::PID << " res = " <<  res << "\n";
  }
  if (i<N-1 && j<M-1) {
    res &= std::equal(&iplusjplus_r[0], &iplusjplus_r[k+i+j+2], &res_iplusjplus_r[0]);
    file << gridtools::PID << " res = " <<  res << "\n";
  }

  if (res)
    file << "RES is TRUE\n";
  else
    file << "RES is FALSE\n";

  int final;
  MPI_Reduce(&res, &final, 1, MPI_INT, MPI_LAND, 0, gridtools::GCL_WORLD);

  if (gridtools::PID==0) {
    if (!final) {
      file << "@" << gridtools::PID << "@ FAILED!\n";
    } else
      file << "@" << gridtools::PID << "@ PASSED!\n";
  }

  MPI_Barrier(gridtools::GCL_WORLD);

  MPI_Finalize();

  return !final;
}
