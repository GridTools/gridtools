
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
#include <sstream>
#include <fstream>
#include <halo_exchange.h>
#include <string>
#include <stdlib.h>
#include <utils/layout_map.h>
#include <utils/boollist.h>
#include <vector>

/** \file Example of use of halo_exchange pattern for regular
    grids. The comments in the code aim at highlight the process of
    instantiating and running a halo exchange pattern.
*/

inline int modulus(int __i, int __j) {
  return (((((__i%__j)<0)?(__j+__i%__j):(__i%__j))));
}

/* This is the data type of the elements of the data arrays.
 */
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

/* Just and utility to print values
 */
void printbuff(std::ostream &file, pair_t* a, int d1, int d2) {
  if (d1<20 && d2<20) {
    file << "------------\n";
    for (int ii=0; ii<d1; ++ii) {
      file << "|";
      for (int jj=0; jj<d2; ++jj) {
        file << a[ii*d2+jj];
      }
      file << "|\n";
    }
    file << "------------\n\n";
  }
}



int main(int argc, char** argv) {

  /* this example is based on MPI Cart Communicators, so we need to
  initialize MPI. This can be done by GCL automatically
  */
  MPI_Init(&argc, &argv);

  /* Now let us initialize GCL itself. If MPI is not initialized at
     this point, it will initialize it
   */
  GCL::GCL_Init(argc, argv);


  /* Each process will hold a tile of size (DIM1+2*H)x(DIM2+2*H). The
     DIM1xDIM2 area inside the H width border is the inner region of
     an hypothetical stencil computation whise halo width is H.
   */
  int DIM1=atoi(argv[1]);
  int DIM2=atoi(argv[2]);
  int H   =atoi(argv[3]);

  /* This example will exchange 3 data arrays at the same time with
     different values.
   */
  pair_t *a = new pair_t[(DIM1+2*H)*(DIM2+2*H)];
  pair_t *b = new pair_t[(DIM1+2*H)*(DIM2+2*H)];
  pair_t *c = new pair_t[(DIM1+2*H)*(DIM2+2*H)];

  /* Just an initialization */
  for (int ii=0; ii<DIM1+2*H; ++ii)
    for (int jj=0; jj<DIM2+2*H; ++jj) {
      a[ii*(DIM2+2*H)+jj] = pair_t(0,0);
      b[ii*(DIM2+2*H)+jj] = pair_t(0,0);                                      
      c[ii*(DIM2+2*H)+jj] = pair_t(0,0);
    }      


  /* Here we compute the computing gris as in many applications
   */
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
  int period[2] = {1, 1};

  file << "@" << pid << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << "\n";
 
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, false, &CartComm);
  int coords[2]={0,0};
  MPI_Cart_get(CartComm, 2, dims, period, coords);


  /* The pattern type is defined with the layouts, data types and
     number of dimensions.

     The logical assumption done in the program is that 'i' is the
     first dimension (rows), and 'j' is the second. The first layout
     states that 'i' is the second dimension in order of strides,
     while 'j' is the first.

     The second layout states that the first dimension in data ('i')
     identify also the first dimension in the communicator. Logically,
     moving on 'i' dimension from processot (p,q) will lead you
     logically to processor (p+1,q).
   */
  typedef GCL::halo_exchange_dynamic_ut<GCL::layout_map<1,0>, 
    GCL::layout_map<0,1>, pair_t, 2, GCL::gcl_cpu, 1 > pattern_type;


  /* The pattern is now instantiated with the periodicities and the
     communicator. The periodicity of the communicator is
     irrelevant. Setting it to be periodic is the best choice, then
     GCL can deal with any periodicity easily.
   */
  pattern_type he(pattern_type::grid_type::period_type(true, true), CartComm);


  /* Next we need to describe the data arrays in terms of halo
     descriptors (see the manual). The 'order' of registration, that
     is the index written within <.> follows the logical order of the
     application. That is, 0 is associated to 'i', while '1' is
     associated to 'j'.
   */
  he.add_halo<0>(H, H, H, DIM1+H-1, DIM1+2*H);
  he.add_halo<1>(H, H, H, DIM2+H-1, DIM2+2*H);

  /* Pattern is set up. This must be done only once per pattern. The
     parameter must me greater or equal to the largest number of
     arrays updated in a single step.
   */
  he.setup(3);


  file << "Proc: (" << coords[0] << ", " << coords[1] << ")\n";


  /* Data is initialized in the inner region of size DIM1xDIM2
   */
  for (int ii=H; ii<DIM1+H; ++ii)
    for (int jj=H; jj<DIM2+H; ++jj) {
      a[ii*(DIM2+2*H)+jj] = pair_t(ii-H+(DIM1)*coords[0],jj-H+(DIM2)*coords[1]);
      b[ii*(DIM2+2*H)+jj] = pair_t(ii-H+(DIM1)*coords[0]+1,jj-H+(DIM2)*coords[1]+1);
      c[ii*(DIM2+2*H)+jj] = pair_t(ii-H+(DIM1)*coords[0]+100,jj-H+(DIM2)*coords[1]+100);
    }

  //printbuff(file,a, DIM1+2*H, DIM2+2*H);
  //   printbuff(file,b, DIM1+2*H, DIM2+2*H);
  printbuff(file,c, DIM1+2*H, DIM2+2*H);
  
  /* This is self explanatory now
   */
  std::vector<pair_t*> vect(3);
  vect[0] = a;
  vect[1] = b;
  vect[2] = c;

  he.pack(vect);

  he.exchange();

  he.unpack(vect);

  file << "\n********************************************************************************\n";

  //printbuff(file,a, DIM1+2*H, DIM2+2*H);
  //   printbuff(file,b, DIM1+2*H, DIM2+2*H);
  printbuff(file,c, DIM1+2*H, DIM2+2*H);

  int passed = true;


  /* Checking the data arrived correctly in the whole region
   */
  for (int ii=0; ii<DIM1+2*H; ++ii)
    for (int jj=0; jj<DIM2+2*H; ++jj) {
      if (a[ii*(DIM2+2*H)+jj] != 
          pair_t(modulus(ii-H+(DIM1)*coords[0], DIM1*dims[0]),
                 modulus(jj-H+(DIM2)*coords[1], DIM2*dims[1])) ) {
        passed = false;
        file << "a " << a[ii*(DIM2+2*H)+jj] << " != " 
             << pair_t(modulus(ii-H+(DIM1)*coords[0], DIM1*dims[0]),
                       modulus(jj-H+(DIM2)*coords[1], DIM2*dims[1])) 
             << "\n";
      }

      if (b[ii*(DIM2+2*H)+jj] != 
          pair_t(modulus(ii-H+(DIM1)*coords[0], DIM1*dims[0])+1,
                 modulus(jj-H+(DIM2)*coords[1], DIM2*dims[1])+1) ) {
        passed = false;
        file << "b " << b[ii*(DIM2+2*H)+jj] << " != " 
             << pair_t(modulus(ii-H+(DIM1)*coords[0], DIM1*dims[0])+1,
                       modulus(jj-H+(DIM2)*coords[1], DIM2*dims[1])+1) 
             << "\n";
      }

      if (c[ii*(DIM2+2*H)+jj] != 
          pair_t(modulus(ii-H+(DIM1)*coords[0], DIM1*dims[0])+100,
                 modulus(jj-H+(DIM2)*coords[1], DIM2*dims[1])+100) ) {
        passed = false;
        file << "c " << c[ii*(DIM2+2*H)+jj] << " != " 
             << pair_t(modulus(ii-H+(DIM1)*coords[0], DIM1*dims[0])+100,
                       modulus(jj-H+(DIM2)*coords[1], DIM2*dims[1])+100) 
             << "\n";
      }
    }

  if (passed)
    file << "RESULT: PASSED!\n";
  else
    file << "RESULT: FAILED!\n";

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
