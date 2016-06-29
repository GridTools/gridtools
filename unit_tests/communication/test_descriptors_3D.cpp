/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include <communication/GCL.hpp>
#include <mpi.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <communication/high-level/descriptors.hpp>
#include <string>
#include <common/boollist.hpp>

#define DIM 10

using gridtools::uint_t;
using gridtools::int_t;

struct triple_t {
  uint_t x,y,z;
  triple_t(uint_t a, uint_t b, uint_t c): x(a), y(b), z(c) {}
  triple_t(): x(-1), y(-1), z(-1) {}
};

std::ostream& operator<<(std::ostream &s, triple_t const & t) {
  return s << " ("
           << t.x << ", "
           << t.y << ", "
           << t.z << ") ";
}

bool operator==(triple_t const & a, triple_t const & b) {
  return (a.x == b.x &&
          a.y == b.y &&
          a.z == b.z);
}

bool operator!=(triple_t const & a, triple_t const & b) {
  return !(a==b);
}


int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  gridtools::GCL_Init(argc, argv);

  triple_t *a = new triple_t[DIM*DIM*DIM];
  triple_t *b = new triple_t[DIM*DIM*DIM];
  triple_t *c = new triple_t[DIM*DIM*DIM];

  uint_t I;

  int pid;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  std::cout << pid << " " << nprocs << "\n";

  std::stringstream ss;
  ss << pid;

  std::string filename = "out" + ss.str() + ".txt";
  //filename[3] = '0'+pid;
  std::cout << filename << std::endl;
  std::ofstream file(filename.c_str());

  file << pid << "  " << nprocs << "\n";

  MPI_Comm CartComm;
  int dims[3] = {0,0,0};
  MPI_Dims_create(nprocs, 3, dims);
  int period[3] = {0, 0, 0};

  file << "@" << pid << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << " - " << dims[2] << "\n";

  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, period, false, &CartComm);

  typedef gridtools::MPI_3D_process_grid_t<3> grid_type;

  gridtools::hndlr_descriptor_ut<triple_t,3,
    gridtools::Halo_Exchange_3D<grid_type> > hd(gridtools::boollist<3>(false,false,false),CartComm);

  I = hd.register_field(a);
  hd.register_halo(I, 2, 2, 1, 3, 6, DIM);
  hd.register_halo(I, 1, 2, 1, 3, 6, DIM);
  hd.register_halo(I, 0, 2, 1, 3, 6, DIM);
  I = hd.register_field(b);
  hd.register_halo(I, 2, 3, 2, 3, 6, DIM);
  hd.register_halo(I, 1, 2, 1, 3, 6, DIM);
  hd.register_halo(I, 0, 3, 2, 4, 6, DIM);
  I = hd.register_field(c);
  hd.register_halo(I, 2, 0, 2, 0, 6, DIM);
  hd.register_halo(I, 1, 3, 2, 3, 6, DIM);
  hd.register_halo(I, 0, 0, 0, 0, 9, DIM);


  int pi, pj, pk;
  hd.pattern().proc_grid().coords(pk, pj, pi);
  int PI, PJ, PK;
  hd.pattern().proc_grid().dims(PK, PJ, PI);

  file << "COORDINATES " << pi << ", " << pj << ", " << pk << std::endl;

  for (uint_t ii=3; ii<=6; ++ii)
    for (uint_t jj=3; jj<=6; ++jj)
      for (uint_t kk=3; kk<=6; ++kk) {
        a[gridtools::access(kk,jj,ii,DIM,DIM,DIM)] = triple_t(ii-3+4*pi,jj-3+4*pj,kk-3+4*pk);
    }

  for (uint_t ii=3; ii<=6; ++ii)
    for (uint_t jj=3; jj<=6; ++jj)
      for (uint_t kk=4; kk<=6; ++kk) {
        b[gridtools::access(kk,jj,ii,DIM,DIM,DIM)] = triple_t(ii-3+4*pi,jj-3+4*pj,kk-4+3*pk);
    }

  for (uint_t ii=0; ii<=6; ++ii)
    for (uint_t jj=3; jj<=6; ++jj)
      for (uint_t kk=0; kk<=9; ++kk) {
        c[gridtools::access(kk,jj,ii,DIM,DIM,DIM)] = triple_t(ii+7*pi,jj-3+4*pj,kk+DIM*pk);
    }

  hd.allocate_buffers();

  hd.pack();

  hd.exchange();

  hd.unpack();

  // CHECK!
  bool err=false;
  for (uint_t ii=3-((pi>0)?2:0); ii<=6+((pi<PI-1)?1:0); ++ii)
    for (uint_t jj=3-((pj>0)?2:0); jj<=6+((pj<PJ-1)?1:0); ++jj)
      for (uint_t kk=3-((pk>0)?2:0); kk<=6+((pk<PK-1)?1:0); ++kk) {
        if (a[gridtools::access(kk,jj,ii,DIM,DIM,DIM)] != triple_t(ii-3+4*pi,jj-3+4*pj,kk-3+4*pk)) {
          err=true;
          file << " A "
                    << ii << ", "
                    << jj << ", "
                    << kk << ", "
                    << a[gridtools::access(kk,jj,ii,DIM,DIM,DIM)] << " != "
                    << triple_t(ii-3+4*pi,jj-3+4*pj,kk-3+4*pk) << "\n";
        }
    }

  for (uint_t ii=3-((pi>0)?3:0); ii<=6+((pi<PI-1)?2:0); ++ii)
    for (uint_t jj=3-((pj>0)?2:0); jj<=6+((pj<PJ-1)?1:0); ++jj)
      for (uint_t kk=4-((pk>0)?3:0); kk<=6+((pk<PK-1)?2:0); ++kk) {
        if (b[gridtools::access(kk,jj,ii,DIM,DIM,DIM)] != triple_t(ii-3+4*pi,jj-3+4*pj,kk-4+3*pk)) {
          err=true;
          file << " B "
                    << ii << ", "
                    << jj << ", "
                    << kk << ", "
                    << b[gridtools::access(kk,jj,ii,DIM,DIM,DIM)] << " != "
                    << triple_t(ii-3+4*pi,jj-3+4*pj,kk-4+3*pk) << "\n";
        }
    }

  for (uint_t ii=0-((pi>0)?0:0); ii<=6+((pi<PI-1)?2:0); ++ii)
    for (uint_t jj=3-((pj>0)?3:0); jj<=6+((pj<PJ-1)?2:0); ++jj)
      for (uint_t kk=0-((pk>0)?0:0); kk<=9+((pk<PK-1)?0:0); ++kk) {
        if (c[gridtools::access(kk,jj,ii,DIM,DIM,DIM)] != triple_t(ii+7*pi,jj-3+4*pj,kk+DIM*pk)) {
          err=true;
          file << " C "
                    << ii << ", "
                    << jj << ", "
                    << kk << ", "
                    << c[gridtools::access(kk,jj,ii,DIM,DIM,DIM)] << " != "
                    << triple_t(ii+7*pi,jj-3+4*pj,kk+DIM*pk) << "\n";
        }
    }


  std::cout << std::boolalpha << err << " (False is good)" << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
