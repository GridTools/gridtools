
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
#include <descriptors_dt.h>
#include <descriptors.h>
#include <string>

#define DIM 10

struct pair_t {
  int x,y;
  pair_t(int a, int b): x(a), y(b) {}
  pair_t(): x(0), y(0) {}
};

std::ostream& operator<<(std::ostream &s, pair_t const & t) { 
  return s << " (" 
           << t.x << ", "
           << t.y << ") ";
}

bool operator==(pair_t const & a, pair_t const & b) {
  return (a.x == b.x && 
          a.y == b.y);
}

bool operator!=(pair_t const & a, pair_t const & b) {
  return !(a==b);
}

void printbuff(std::ostream &file, pair_t* a) {
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

  pair_t *a = new pair_t[DIM*DIM*DIM];
  pair_t *b = new pair_t[DIM*DIM*DIM];
  pair_t *c = new pair_t[DIM*DIM*DIM];

  for (int ii=0; ii<=DIM; ++ii)
    for (int jj=0; jj<=DIM; ++jj) {
      a[GCL::access(jj,ii,DIM,DIM)] = pair_t(0,0);
      b[GCL::access(jj,ii,DIM,DIM)] = pair_t(0,0);                                      
      c[GCL::access(jj,ii,DIM,DIM)] = pair_t(0,0);
    }      

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

  typedef GCL::_2D_process_grid_t<GCL::gcl_utils::boollist<2> > grid_type;

  GCL::hndlr_dynamic_ut<pair_t,2, 
    GCL::Halo_Exchange_2D<grid_type>
    > hd(GCL::gcl_utils::boollist<2>(false,false), nprocs, pid);

  hd.halo.add_halo(1, 2, 1, 3, 6, DIM);
  hd.halo.add_halo(0, 2, 1, 3, 6, DIM);

  hd.allocate_buffers(3);

  int pi, pj;
  hd.pattern().proc_grid().coords(pi, pj);
  int PI, PJ;
  hd.pattern().proc_grid().dims(PI, PJ);

  file << "Proc: (" << pi << ", " << pj << ")\n";


  for (int ii=3; ii<=6; ++ii)
    for (int jj=3; jj<=6; ++jj) {
      // if (pid==6)
      //   a[GCL::access(jj,ii,DIM,DIM)] = pair_t(-(ii-3+4*pj),-(jj-3+4*pi));
      // else
        a[GCL::access(jj,ii,DIM,DIM)] = pair_t(ii-3+4*pj,jj-3+4*pi);
    }      

  for (int ii=3; ii<=6; ++ii)
    for (int jj=3; jj<=6; ++jj) {
        b[GCL::access(jj,ii,DIM,DIM)] = pair_t(ii-3+4*pj,jj-3+4*pi);
    }      

  for (int ii=3; ii<=6; ++ii)
    for (int jj=3; jj<=6; ++jj) {
        c[GCL::access(jj,ii,DIM,DIM)] = pair_t(ii-3+4*pj,jj-3+4*pi);
    }      

  printbuff(file,a);
  printbuff(file,b);
  printbuff(file,c);

  hd.pack(a,b,c);

  hd.exchange();

  hd.unpack(a,b,c);

  file << "\n********************************************************************************\n";

  printbuff(file,a);
  printbuff(file,b);
  printbuff(file,c);

  // CHECK!
  bool err=false;
  for (int ii=3-((pj>0)?2:0); ii<=6+((pj<PJ-1)?1:0); ++ii)
    for (int jj=3-((pi>0)?2:0); jj<=6+((pi<PI-1)?1:0); ++jj) {
        if (a[GCL::access(jj,ii,DIM,DIM)] != pair_t(ii-3+4*pj,jj-3+4*pi)) {
          err=true;
          file << " A " 
                    << ii << ", "
                    << jj << ", "
                    << a[GCL::access(jj,ii,DIM,DIM)] << " != "
                    << pair_t(ii-2+4*pj,jj-2+4*pi) << "\n";
        }
    }

  for (int ii=3-((pj>0)?2:0); ii<=6+((pj<PJ-1)?1:0); ++ii)
    for (int jj=3-((pi>0)?2:0); jj<=6+((pi<PI-1)?1:0); ++jj) {
        if (b[GCL::access(jj,ii,DIM,DIM)] != pair_t(ii-3+4*pj,jj-3+4*pi)) {
          err=true;
          file << " B "
                    << ii << ", "
                    << jj << ", "
                    << b[GCL::access(jj,ii,DIM,DIM)] << " != "
                    << pair_t(ii-3+4*pj,jj-3+4*pi) << "\n";
        }
    }

  for (int ii=3-((pj>0)?2:0); ii<=6+((pj<PJ-1)?1:0); ++ii)
    for (int jj=3-((pi>0)?2:0); jj<=6+((pi<PI-1)?1:0); ++jj) {
        if (c[GCL::access(jj,ii,DIM,DIM)] != pair_t(ii-3+4*pj,jj-3+4*pi)) {
          err=true;
          file << " C "
                    << ii << ", "
                    << jj << ", "
                    << c[GCL::access(jj,ii,DIM,DIM)] << " != "
                    << pair_t(ii+7*pj,jj-3+4*pi) << "\n";
        }
    }


  std::cout << std::boolalpha << err << " (False is good)" << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
