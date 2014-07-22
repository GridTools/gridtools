#include <communication/low-level/proc_grids_2D.h>
#include <communication/all_to_all_halo.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <common/array.h>
#include <common/boollist.h>

template <typename STREAM, typename T>
void print(STREAM & cout, std::vector<T> const& v, int n, int m) {
  if ((n < 20) && (m < 20)) {
    for (int i=0; i<n; ++i) {
      cout << "@" << gridtools::PID << "@ ";
      for (int j=0; j<m; ++j) {
        cout << v[i*m+j] << " ";
      }
      cout << "\n";
    }
    cout << "\n\n";
  }
}


int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);
  gridtools::GCL_Init(argc, argv);

  std::stringstream ss;
  ss << gridtools::PID;

  std::string filename = "out" + ss.str() + ".txt";
  //filename[3] = '0'+pid;
  std::cout << filename << std::endl;
  std::ofstream file(filename.c_str());

  typedef gridtools::array<gridtools::halo_descriptor, 2> halo_block;;

  typedef gridtools::_2D_process_grid_t<gridtools::boollist<2> > grid_type;

  grid_type pgrid(gridtools::boollist<2>(true,true), gridtools::PROCS, gridtools::PID);

  gridtools::all_to_all_halo<int, grid_type> a2a(pgrid);

  int pi, pj;
  int PI, PJ;
  pgrid.coords(pi,pj);
  pgrid.dims(PI,PJ);

  file << "@" << gridtools::PID << "@ PROC GRID SIZE " << PI << "x" << PJ << "\n";

  if (argc != 3) {
    std::cout << "Usage: pass two arguments: tile size (edge) followed by the halo width\n";
    return 1;
  }

  int N,H;
  N = atoi(argv[1]);
  H = atoi(argv[2]);


  file << "@" << gridtools::PID << "@ PARAMETER N " << N << "\n";

  std::vector<int> dataout(PI*N*PJ*N);
  std::vector<int> datain((N+2*H)*(N+2*H));

  file << "Address of data: " << (void*)(&(dataout[0]))
       << ", data in " << (void*)(&(datain[0])) << "\n";

  gridtools::array<int,2> crds;

  if (gridtools::PID == 0) {
    file << "INITIALIZING DATA TO SEND\n";

    halo_block send_block;

    for(int i=0; i<PI; ++i) {
      for (int j=0; j<PJ; ++j) {

        crds[0] = i;
        crds[1] = j; // INCREASING STRIDES

        // INCREASING STRIDES
        send_block[0] = gridtools::halo_descriptor(0,0,j*N,(j+1)*N-1, PJ*N);
        send_block[1] = gridtools::halo_descriptor(0,0,i*N,(i+1)*N-1, N*PI);

        a2a.register_block_to(&dataout[0], send_block, crds);
      }
    }
  }

  crds[0] = 0;
  crds[1] = 0; // INCREASING STRIDES

  // INCREASING STRIDES
  halo_block recv_block;
  recv_block[0] = gridtools::halo_descriptor(H,H,H,N+H-1, N+2*H);
  recv_block[1] = gridtools::halo_descriptor(H,H,H,N+H-1, N+2*H);

  a2a.register_block_from(&datain[0], recv_block, crds);

  for(int i=0; i<PI*N; ++i)
    for (int j=0; j<PJ*N; ++j) {
      dataout[j*PI*N+i] = i*PJ*N+j;
    }

  for(int i=0; i<N+2*H; ++i)
    for (int j=0; j<N+2*H; ++j) {
      datain[i*N+j] = 0;
    }

  print(file, dataout, PI*N, PJ*N);
  print(file, datain, N+2*H, N+2*H);
  file.flush();

  MPI_Barrier(gridtools::GCL_WORLD);

  a2a.setup();
  a2a.start_exchange();
  a2a.wait();

  print(file, dataout, PI*N, PJ*N);
  print(file, datain, N+2*H, N+2*H);

  bool correct = true;
  for(int i=H; i<N+H; ++i)
    for (int j=H; j<N+H; ++j) {
      // (dataout[(i-H+N*pi)*N*PJ+(j-H+N*pj)] != datain[i*N+j])
      // if (dataout[(i-H+N*pi)*PJ+(j-H+N*pj)] != datain[i*(N+2*H)+j]) {
      if (dataout[(i-H+N*pi)*N*PJ+(j-H+N*pj)] != datain[i*(N+2*H)+j]) {
        file << "(" << i << "," << j
             << ") (" << i-H+N*pi << "," << j-H+N*pj
             << ") Expected " << dataout[(i-H+N*pj)*N*PI+(j-H+N*pj)] << " got " << datain[i*(N+2*H)+j] << std::endl;
        correct = false;
      }
    }

  file << "RESULT: ";
  if (correct) {
    file << "PASSED!\n";
  } else {
    file << "FAILED!\n";
  }

  file.flush();
  file.close();

  MPI_Barrier(gridtools::GCL_WORLD);

  gridtools::GCL_Finalize();

  return 0;
}
