
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


#include <proc_grids_2D.h>
#include <Generic_All_to_All.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>


template <typename STREAM, typename T>
void print(STREAM & cout, std::vector<T> const& v, int n, int m) {
  if ((n < 20) && (m < 20)) {
    for (int i=0; i<n; ++i) {
      cout << "@" << GCL::PID << "@ ";
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
  GCL::GCL_Init(argc, argv);

  std::stringstream ss;
  ss << GCL::PID;

  std::string filename = "out" + ss.str() + ".txt";
  //filename[3] = '0'+pid;
  std::cout << filename << std::endl;
  std::ofstream file(filename.c_str());

  GCL::all_to_all<int> a2a(GCL::PROCS);

  // Let's do a 2D scatter
  // Step 1: define data
  typedef GCL::_2D_process_grid_t<GCL::gcl_utils::boollist<2> > grid_type;

  grid_type pgrid(GCL::gcl_utils::boollist<2>(true,true), GCL::PROCS, GCL::PID);

  int pi, pj;
  int PI, PJ;
  pgrid.coords(pi,pj);
  pgrid.dims(PI,PJ);

  file << "@" << GCL::PID << "@ PROC GRID SIZE " << PI << "x" << PJ << "\n";

  int N = atoi(argv[1]);

  file << "@" << GCL::PID << "@ PARAMETER N " << N << "\n";

  std::vector<int> dataout(PI*N*PJ*N);
  std::vector<int> datain(N*N);

  file << "Address of data: " << (void*)(&(dataout[0])) 
       << ", data in " << (void*)(&(datain[0])) << "\n";

  if (GCL::PID == 0) {
    file << "INITIALIZING DATA TO SEND\n";
    std::vector<int> out_sizes(2), out_subsizes(2), out_starts(2);
    std::vector<MPI_Datatype> DATAOUT(PI*PJ) ;

    out_sizes[0] = N*PI;
    out_sizes[1] = N*PJ;
    out_subsizes[0] = N;
    out_subsizes[1] = N;

    for(int i=0; i<PI; ++i) {
      for (int j=0; j<PJ; ++j) {
        int k=i*PJ+j;
        out_starts[0] = i*N;
        out_starts[1] = j*N;
        MPI_Type_create_subarray(2, 
                                 &(out_sizes[0]), 
                                 &(out_subsizes[0]), 
                                 &(out_starts[0]), 
                                 MPI_ORDER_C,
                                 MPI_INT,
                                 &(DATAOUT[k]));
        MPI_Type_commit(&(DATAOUT[k]));
      }
    }

  
    for(int i=0; i<PI; ++i)
      for (int j=0; j<PJ; ++j) {
        int k=i*PJ+j;
        file << "Setting a2a " << k << "\n";
        a2a.to[k] = GCL::packet<int>(DATAOUT[k], &dataout[0]);
      }
  } else {
    for(int i=0; i<PI; ++i)
      for (int j=0; j<PJ; ++j) {
        int k=i*PJ+j;
        file << "Setting a2a " << k << " to null\n";
        a2a.to[k] = GCL::packet<int>(); // set pointer to NULL
      }
  }

  std::vector<int> in_sizes(2), in_subsizes(2), in_starts(2);

  in_sizes[0] = N;
  in_sizes[1] = N;
  in_subsizes[0] = N;
  in_subsizes[1] = N;
  in_starts[0] = 0;
  in_starts[1] = 0;

  MPI_Datatype DATAIN;
  MPI_Type_create_subarray(2, 
                           &in_sizes[0], 
                           &in_subsizes[0], 
                           &in_starts[0], 
                           MPI_ORDER_C,
                           MPI_INT,
                           &DATAIN);
  MPI_Type_commit(&DATAIN);
  

  for(int i=0; i<PI; ++i)
    for (int j=0; j<PJ; ++j) {
      int k=i*PJ+j;
      a2a.from[k] = GCL::packet<int>();
    }

  a2a.from[0] = GCL::packet<int>(DATAIN, &datain[0]);

  for(int i=0; i<PI*N; ++i)
    for (int j=0; j<PJ*N; ++j) {
      dataout[i*PJ*N+j] = i*PJ*N+j;
    }
  
  for(int i=0; i<N; ++i)
    for (int j=0; j<N; ++j) {
      datain[i*N+j] = 0;
    }

  print(file, dataout, PI*N, PJ*N);
  print(file, datain, N, N);
  file.flush();

  MPI_Barrier(GCL::GCL_WORLD);

  a2a.setup();
  a2a.start_exchange();
  a2a.wait();

  print(file, dataout, PI*N, PJ*N);
  print(file, datain, N, N);

  bool correct = true;
  for(int i=0; i<N; ++i)
    for (int j=0; j<N; ++j) {
      if (dataout[(i+N*pi)*N*PJ+(j+N*pj)] != datain[i*N+j])
        correct = false;
    }

  file << "RESULT: ";
  if (correct) {
    file << "PASSED!\n";
  } else {
    file << "FAILED!\n";
  }

  file.flush();
  file.close();

  MPI_Barrier(GCL::GCL_WORLD);

  GCL::GCL_Finalize();

  return 0;
}
