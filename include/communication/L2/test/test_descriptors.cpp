
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


#include <GCL.h>
#include <mpi.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <descriptors.h>
#include <string>
#include <utils/boollist.h>

void printbuff(std::ostream &file, char* a) {
  file << "------------\n";
  for (int ii=0; ii<10; ++ii) {
    file << "|";
    for (int jj=0; jj<10; ++jj) {
      file << a[ii*10+jj];
    }
    file << "|\n";
  }
  file << "------------\n\n";
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  GCL::GCL_Init(argc, argv);

  char *a = new char[100];
  char *b = new char[100];
  char *c = new char[100];


  for (int ii=0; ii<10; ++ii)
    for (int jj=0; jj<10; ++jj) {
      a[ii*10+jj] = '0'+ii;
      b[ii*10+jj] = '0'+ii;
      c[ii*10+jj] = '0'+ii;
    }      

  int I; 

  int pid;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  std::cout << pid << " " << nprocs << "\n";

  std::stringstream ss;
  ss << pid;
  std::string filename = "out" + ss.str() + ".txt";

  std::cout << filename << std::endl;
  std::ofstream file(filename.c_str());

  file << pid << "  " << nprocs << "\n";

  MPI_Comm CartComm;
  int dims[2] = {0,0};
  MPI_Dims_create(nprocs, 2, dims);
  int period[2] = {0, 0};

  file << "@" << pid << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << "\n";
 
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, false, &CartComm);

  typedef GCL::Halo_Exchange_2D<GCL::MPI_2D_process_grid_t<GCL::gcl_utils::boollist<2> > > pattern_type;

  GCL::hndlr_descriptor_ut<char,2, pattern_type> hd(GCL::gcl_utils::boollist<2>(true,true), CartComm);

  I = hd.register_field(a);
  hd.register_halo(I, 0, 2, 1, 3, 6, 10);
  hd.register_halo(I, 1, 2, 1, 3, 6, 10);
  I = hd.register_field(b);
  hd.register_halo(I, 0, 3, 2, 3, 6, 10);
  hd.register_halo(I, 1, 2, 1, 3, 6, 10);
  I = hd.register_field(c);
  hd.register_halo(I, 0, 0, 2, 0, 6, 10);
  hd.register_halo(I, 1, 3, 2, 3, 6, 10);

  for (int ii=3; ii<=6; ++ii)
    for (int jj=3; jj<=6; ++jj)
      a[ii*10+jj] = 'a'+pid*3;

  for (int ii=3; ii<=6; ++ii)
    for (int jj=3; jj<=6; ++jj)
      b[ii*10+jj] = 'b'+pid*3;

  for (int ii=3; ii<=6; ++ii)
    for (int jj=0; jj<=6; ++jj)
      c[ii*10+jj] = 'c'+pid*3;

  printbuff(file,a);
  printbuff(file,b);
  printbuff(file,c);

  file << " ----------------" << "\n"
            << "9| |  |    | |  |" << "\n"
            << "8| |  |    | |  |" << "\n"
            << " |-|--|----|-|--|" << "\n"
            << "7| |ss|tttt|u|  |" << "\n"
            << " |-|--|----|-|--|" << "\n"
            << "6| |vv|    |w|  |" << "\n"
            << "5| |vv|    |w|  |" << "\n"
            << "4| |vv|    |w|  |" << "\n"
            << "3| |vv|    |w|  |" << "\n"
            << " |-|--|----|-|--|" << "\n"
            << "2| |xx|yyyy|z|  |" << "\n"
            << "1| |xx|yyyy|z|  |" << "\n"
            << " |-|--|----|-|--|" << "\n"
            << "0| |  |    | |  |" << "\n"
            << " |-|--|----|-|--|" << "\n"
            << "  0 12 3456 7 89" << "\n\n";


  file << "DF 0 size 0: (-1,-1) x " << hd.data_field(0).send_buffer_size(GCL::gcl_utils::make_array(-1,-1)) << "\n"
            << "DF 0 size 1: (0,-1)  y " << hd.data_field(0).send_buffer_size(GCL::gcl_utils::make_array(0,-1)) << "\n"
            << "DF 0 size 2: (1,-1)  z " << hd.data_field(0).send_buffer_size(GCL::gcl_utils::make_array(1,-1)) << "\n"
            << "DF 0 size 3: (-1,0)  v " << hd.data_field(0).send_buffer_size(GCL::gcl_utils::make_array(-1,0)) << "\n"
            << "DF 0 size 4: (1,0)   w " << hd.data_field(0).send_buffer_size(GCL::gcl_utils::make_array(1,0)) << "\n"
            << "DF 0 size 5: (-1,1)  s " << hd.data_field(0).send_buffer_size(GCL::gcl_utils::make_array(-1,1)) << "\n"
            << "DF 0 size 6: (0,1)   t " << hd.data_field(0).send_buffer_size(GCL::gcl_utils::make_array(0,1)) << "\n"
            << "DF 0 size 7: (1,1)   u " << hd.data_field(0).send_buffer_size(GCL::gcl_utils::make_array(1,1)) << "\n"
            << "\n";

  file << "DF 0 r size 0: (-1,-1) x " << hd.data_field(0).recv_buffer_size(GCL::gcl_utils::make_array(-1,-1)) << "\n"
            << "DF 0 r size 1: (0,-1)  y " << hd.data_field(0).recv_buffer_size(GCL::gcl_utils::make_array(0,-1)) << "\n"
            << "DF 0 r size 2: (1,-1)  z " << hd.data_field(0).recv_buffer_size(GCL::gcl_utils::make_array(1,-1)) << "\n"
            << "DF 0 r size 3: (-1,0)  v " << hd.data_field(0).recv_buffer_size(GCL::gcl_utils::make_array(-1,0)) << "\n"
            << "DF 0 r size 4: (1,0)   w " << hd.data_field(0).recv_buffer_size(GCL::gcl_utils::make_array(1,0)) << "\n"
            << "DF 0 r size 5: (-1,1)  s " << hd.data_field(0).recv_buffer_size(GCL::gcl_utils::make_array(-1,1)) << "\n"
            << "DF 0 r size 6: (0,1)   t " << hd.data_field(0).recv_buffer_size(GCL::gcl_utils::make_array(0,1)) << "\n"
            << "DF 0 r size 7: (1,1)   u " << hd.data_field(0).recv_buffer_size(GCL::gcl_utils::make_array(1,1)) << "\n"
            << "\n";

  file << " ---------------" << "\n"
            << "9|   |    |  | |" << "\n"
            << "8|   |    |  | |" << "\n"
            << " |---|----|--|-|" << "\n"
            << "7|sss|tttt|uu| |" << "\n"
            << " |---|----|--|-|" << "\n"
            << "6|vvv|    |ww| |" << "\n"
            << "5|vvv|    |ww| |" << "\n"
            << "4|vvv|    |ww| |" << "\n"
            << "3|vvv|    |ww| |" << "\n"
            << " |---|----|--|-|" << "\n"
            << "2|xxx|yyyy|zz| |" << "\n"
            << "1|xxx|yyyy|zz| |" << "\n"
            << " |---|----|--|-|" << "\n"
            << "0|   |    |  | |" << "\n"
            << " |---|----|--|-|" << "\n"
            << "  012 3456 78 9" << "\n\n";


  file << "DF 1 size 0: (-1,-1) x " << hd.data_field(1).send_buffer_size(GCL::gcl_utils::make_array(-1,-1)) << "\n"
            << "DF 1 size 1: (0,-1)  y " << hd.data_field(1).send_buffer_size(GCL::gcl_utils::make_array(0,-1)) << "\n"
            << "DF 1 size 2: (1,-1)  z " << hd.data_field(1).send_buffer_size(GCL::gcl_utils::make_array(1,-1)) << "\n"
            << "DF 1 size 3: (-1,0)  v " << hd.data_field(1).send_buffer_size(GCL::gcl_utils::make_array(-1,0)) << "\n"
            << "DF 1 size 4: (1,0)   w " << hd.data_field(1).send_buffer_size(GCL::gcl_utils::make_array(1,0)) << "\n"
            << "DF 1 size 5: (-1,1)  s " << hd.data_field(1).send_buffer_size(GCL::gcl_utils::make_array(-1,1)) << "\n"
            << "DF 1 size 6: (0,1)   t " << hd.data_field(1).send_buffer_size(GCL::gcl_utils::make_array(0,1)) << "\n"
            << "DF 1 size 7: (1,1)   u " << hd.data_field(1).send_buffer_size(GCL::gcl_utils::make_array(1,1)) << "\n"
            << "\n";

  file << "DF 1 r size 0: (-1,-1) x " << hd.data_field(1).recv_buffer_size(GCL::gcl_utils::make_array(-1,-1)) << "\n"
            << "DF 1 r size 1: (0,-1)  y " << hd.data_field(1).recv_buffer_size(GCL::gcl_utils::make_array(0,-1)) << "\n"
            << "DF 1 r size 2: (1,-1)  z " << hd.data_field(1).recv_buffer_size(GCL::gcl_utils::make_array(1,-1)) << "\n"
            << "DF 1 r size 3: (-1,0)  v " << hd.data_field(1).recv_buffer_size(GCL::gcl_utils::make_array(-1,0)) << "\n"
            << "DF 1 r size 4: (1,0)   w " << hd.data_field(1).recv_buffer_size(GCL::gcl_utils::make_array(1,0)) << "\n"
            << "DF 1 r size 5: (-1,1)  s " << hd.data_field(1).recv_buffer_size(GCL::gcl_utils::make_array(-1,1)) << "\n"
            << "DF 1 r size 6: (0,1)   t " << hd.data_field(1).recv_buffer_size(GCL::gcl_utils::make_array(0,1)) << "\n"
            << "DF 1 r size 7: (1,1)   u " << hd.data_field(1).recv_buffer_size(GCL::gcl_utils::make_array(1,1)) << "\n"
            << "\n";

  file << " --------------" << "\n"
            << "9|       |  | |" << "\n"
            << " |-------|--|-|" << "\n"
            << "8|ttttttt|uu| |" << "\n"
            << "7|ttttttt|uu| |" << "\n"
            << " |-------|--|-|" << "\n"
            << "6|       |ww| |" << "\n"
            << "5|       |ww| |" << "\n"
            << "4|       |ww| |" << "\n"
            << "3|       |ww| |" << "\n"
            << " |-------|--|-|" << "\n"
            << "2|yyyyyyy|zz| |" << "\n"
            << "1|yyyyyyy|zz| |" << "\n"
            << "0|yyyyyyy|zz| |" << "\n"
            << " |-------|--|-|" << "\n"
            << "  0123456 78 9" << "\n\n";


  file << "DF 2 size 0: (-1,-1) x " << hd.data_field(2).send_buffer_size(GCL::gcl_utils::make_array(-1,-1)) << "\n"
            << "DF 2 size 1: (0,-1)  y " << hd.data_field(2).send_buffer_size(GCL::gcl_utils::make_array(0,-1)) << "\n"
            << "DF 2 size 2: (1,-1)  z " << hd.data_field(2).send_buffer_size(GCL::gcl_utils::make_array(1,-1)) << "\n"
            << "DF 2 size 3: (-1,0)  v " << hd.data_field(2).send_buffer_size(GCL::gcl_utils::make_array(-1,0)) << "\n"
            << "DF 2 size 4: (1,0)   w " << hd.data_field(2).send_buffer_size(GCL::gcl_utils::make_array(1,0)) << "\n"
            << "DF 2 size 5: (-1,1)  s " << hd.data_field(2).send_buffer_size(GCL::gcl_utils::make_array(-1,1)) << "\n"
            << "DF 2 size 6: (0,1)   t " << hd.data_field(2).send_buffer_size(GCL::gcl_utils::make_array(0,1)) << "\n"
            << "DF 2 size 7: (1,1)   u " << hd.data_field(2).send_buffer_size(GCL::gcl_utils::make_array(1,1)) << "\n"
            << "\n";

  file << "DF 2 r size 0: (-1,-1) x " << hd.data_field(2).recv_buffer_size(GCL::gcl_utils::make_array(-1,-1)) << "\n"
            << "DF 2 r size 1: (0,-1)  y " << hd.data_field(2).recv_buffer_size(GCL::gcl_utils::make_array(0,-1)) << "\n"
            << "DF 2 r size 2: (1,-1)  z " << hd.data_field(2).recv_buffer_size(GCL::gcl_utils::make_array(1,-1)) << "\n"
            << "DF 2 r size 3: (-1,0)  v " << hd.data_field(2).recv_buffer_size(GCL::gcl_utils::make_array(-1,0)) << "\n"
            << "DF 2 r size 4: (1,0)   w " << hd.data_field(2).recv_buffer_size(GCL::gcl_utils::make_array(1,0)) << "\n"
            << "DF 2 r size 5: (-1,1)  s " << hd.data_field(2).recv_buffer_size(GCL::gcl_utils::make_array(-1,1)) << "\n"
            << "DF 2 r size 6: (0,1)   t " << hd.data_field(2).recv_buffer_size(GCL::gcl_utils::make_array(0,1)) << "\n"
            << "DF 2 r size 7: (1,1)   u " << hd.data_field(2).recv_buffer_size(GCL::gcl_utils::make_array(1,1)) << "\n"
            << "\n";

  for (int jj=-1; jj<=1; ++jj)
    for (int ii=-1; ii<=1; ++ii)
      file << "Total pack size   " << ii << ", " << jj << " = " << hd.total_pack_size(GCL::gcl_utils::make_array(ii,jj)) << "\n";

  for (int jj=-1; jj<=1; ++jj)
    for (int ii=-1; ii<=1; ++ii)
      file << "Total unpack size " << ii << ", " << jj << " = " << hd.total_unpack_size(GCL::gcl_utils::make_array(ii,jj)) << "\n";

  hd.allocate_buffers();

  hd.pack();

  hd.exchange();

  hd.unpack();

  printbuff(file,a);
  printbuff(file,b);
  printbuff(file,c);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
