
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

/** \file Example of use of halo_exchange pattern for regular
    grids. The comments in the code aim at highlight the process of
    instantiating and running a halo exchange pattern.
*/

inline int modulus(int __i, int __j) {
  return (((((__i%__j)<0)?(__j+__i%__j):(__i%__j))));
}

/* This is the data type of the elements of the data arrays.
 */
struct triple_t {
  int x,y,z;
  triple_t(int a, int b, int c): x(a), y(b), z(c) {}
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

/* Just and utility to print values
 */
void printbuff(std::ostream &file, triple_t* a, int d1, int d2, int d3) {
  if (d1<6 && d2<6 && d3<6) {
    file << "------------\n";
    for (int ii=0; ii<d1; ++ii) {
      file << "|";
      for (int jj=0; jj<d2; ++jj) {
        for (int kk=0; kk<d2; ++kk) {
          file << a[d1*d2*kk+ii*d2+jj];
        }
        file << "|\n";
      }
      file << "\n\n";
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


  /* Each process will hold a tile of size
     (DIM1+2*H)x(DIM2+2*H)x(DIM3+2*H). The DIM1xDIM2xDIM3 area inside
     the H width border is the inner region of an hypothetical stencil
     computation whise halo width is H.
   */
  int DIM1=atoi(argv[1]);
  int DIM2=atoi(argv[2]);
  int DIM3=atoi(argv[3]);
  int H   =atoi(argv[4]);

  /* This example will exchange 3 data arrays at the same time with
     different values.
   */
  triple_t *a = new triple_t[(DIM1+2*H)*(DIM2+2*H)*(DIM3+2*H)];
  triple_t *b = new triple_t[(DIM1+2*H)*(DIM2+2*H)*(DIM3+2*H)];
  triple_t *c = new triple_t[(DIM1+2*H)*(DIM2+2*H)*(DIM3+2*H)];

  /* Just an initialization */
  for (int ii=0; ii<DIM1+2*H; ++ii)
    for (int jj=0; jj<DIM2+2*H; ++jj) {
      for (int kk=0; kk<DIM3+2*H; ++kk) {
        a[(DIM1+2*H)*(DIM2+2*H)*kk+ii*(DIM2+2*H)+jj] = triple_t(0,0,0);
        b[(DIM1+2*H)*(DIM2+2*H)*kk+ii*(DIM2+2*H)+jj] = triple_t(0,0,0);                                      
        c[(DIM1+2*H)*(DIM2+2*H)*kk+ii*(DIM2+2*H)+jj] = triple_t(0,0,0);
      }
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
  int dims[3] = {0,0,0};
  MPI_Dims_create(nprocs, 3, dims);
  int period[3] = {1, 1, 1};

  file << "@" << pid << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << " - " << dims[2] << "\n";
 
  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, period, false, &CartComm);
  int coords[3]={0,0,0};
  MPI_Cart_get(CartComm, 3, dims, period, coords);


  /* The pattern type is defined with the layouts, data types and
     number of dimensions.

     The logical assumption done in the program is that 'i' is the
     first dimension (rows), 'j' is the second, and 'k' is the
     third. The first layout states that 'i' is the second dimension
     in order of strides, while 'j' is the first and 'k' is the third
     (just by looking at the initialization loops this shoule be
     clear).

     The second layout states that the first dimension in data ('i')
     identify also the first dimension in the communicator. Logically,
     moving on 'i' dimension from processot (p,q,r) will lead you
     logically to processor (p+1,q,r). The other dimensions goes as
     the others.
   */
  typedef GCL::halo_exchange_dynamic_ut<GCL::layout_map<1,0,2>, 
    GCL::layout_map<0,1,2>, triple_t, 3, GCL::gcl_cpu, 1 > pattern_type;


  /* The pattern is now instantiated with the periodicities and the
     communicator. The periodicity of the communicator is
     irrelevant. Setting it to be periodic is the best choice, then
     GCL can deal with any periodicity easily.
  */
  pattern_type he(pattern_type::grid_type::period_type(true, true,true), CartComm);


  /* Next we need to describe the data arrays in terms of halo
     descriptors (see the manual). The 'order' of registration, that
     is the index written within <.> follows the logical order of the
     application. That is, 0 is associated to 'i', '1' is
     associated to 'j', '2' to 'k'.
  */
  he.add_halo<0>(H, H, H, DIM1+H-1, DIM1+2*H);
  he.add_halo<1>(H, H, H, DIM2+H-1, DIM2+2*H);
  he.add_halo<2>(H, H, H, DIM3+H-1, DIM3+2*H);

  /* Pattern is set up. This must be done only once per pattern. The
     parameter must me greater or equal to the largest number of
     arrays updated in a single step.
  */
  he.setup(3);


  file << "Proc: (" << coords[0] << ", " << coords[1] << ", " << coords[2] << ")\n";


  /* Data is initialized in the inner region of size DIM1xDIM2
   */
  for (int ii=H; ii<DIM1+H; ++ii)
    for (int jj=H; jj<DIM2+H; ++jj) 
      for (int kk=H; kk<DIM3+H; ++kk) {
        a[(DIM1+2*H)*(DIM2+2*H)*kk+ii*(DIM2+2*H)+jj] = 
          triple_t(ii-H+(DIM1)*coords[0],
                   jj-H+(DIM2)*coords[1],
                   kk-H+(DIM3)*coords[2]);
        b[(DIM1+2*H)*(DIM2+2*H)*kk+ii*(DIM2+2*H)+jj] = 
          triple_t(ii-H+(DIM1)*coords[0]+1,
                   jj-H+(DIM2)*coords[1]+1,
                   kk-H+(DIM3)*coords[2]+1);
        c[(DIM1+2*H)*(DIM2+2*H)*kk+ii*(DIM2+2*H)+jj] = 
          triple_t(ii-H+(DIM1)*coords[0]+100,
                   jj-H+(DIM2)*coords[1]+100,
                   kk-H+(DIM3)*coords[2]+100);
      }

  printbuff(file,a, DIM1+2*H, DIM2+2*H, DIM3+2*H);
  printbuff(file,b, DIM1+2*H, DIM2+2*H, DIM3+2*H);
  printbuff(file,c, DIM1+2*H, DIM2+2*H, DIM3+2*H);
  
  /* This is self explanatory now
   */
std::vector<triple_t*> vect(3);
vect[0] = a;
vect[1] = b;
vect[2] = c;

  he.pack(vect);

  he.exchange();

  he.unpack(vect);

  file << "\n********************************************************************************\n";

  printbuff(file,a, DIM1+2*H, DIM2+2*H, DIM3+2*H);
  printbuff(file,b, DIM1+2*H, DIM2+2*H, DIM3+2*H);
  printbuff(file,c, DIM1+2*H, DIM2+2*H, DIM3+2*H);

  int passed = true;


  /* Checking the data arrived correctly in the whole region
   */
  for (int ii=0; ii<DIM1+2*H; ++ii)
    for (int jj=0; jj<DIM2+2*H; ++jj)
      for (int kk=0; kk<DIM3+2*H; ++kk) {
        if (a[(DIM1+2*H)*(DIM2+2*H)*kk+ii*(DIM2+2*H)+jj] != 
            triple_t(modulus(ii-H+(DIM1)*coords[0], DIM1*dims[0]),
                     modulus(jj-H+(DIM2)*coords[1], DIM2*dims[1]), 
                     modulus(kk-H+(DIM3)*coords[2], DIM3*dims[2])) 
            ) {
          passed = false;
          file << "a " << a[(DIM1+2*H)*(DIM2+2*H)*kk+ii*(DIM2+2*H)+jj] << " != " 
               << triple_t(modulus(ii-H+(DIM1)*coords[0], DIM1*dims[0]),
                           modulus(jj-H+(DIM2)*coords[1], DIM2*dims[1]), 
                           modulus(kk-H+(DIM3)*coords[2], DIM3*dims[2])) 
               << "\n";
        }

        if (b[(DIM1+2*H)*(DIM2+2*H)*kk+ii*(DIM2+2*H)+jj] != 
            triple_t(modulus(ii-H+(DIM1)*coords[0], DIM1*dims[0])+1,
                     modulus(jj-H+(DIM2)*coords[1], DIM2*dims[1])+1, 
                     modulus(kk-H+(DIM3)*coords[2], DIM3*dims[2])+1) 
            ) {
          passed = false;
          file << "b " << b[(DIM1+2*H)*(DIM2+2*H)*kk+ii*(DIM2+2*H)+jj] << " != " 
               << triple_t(modulus(ii-H+(DIM1)*coords[0], DIM1*dims[0])+1,
                           modulus(jj-H+(DIM2)*coords[1], DIM2*dims[1])+1, 
                           modulus(kk-H+(DIM3)*coords[2], DIM3*dims[2])+1) 
               << "\n";
        }

        if (c[(DIM1+2*H)*(DIM2+2*H)*kk+ii*(DIM2+2*H)+jj] != 
            triple_t(modulus(ii-H+(DIM1)*coords[0], DIM1*dims[0])+100,
                     modulus(jj-H+(DIM2)*coords[1], DIM2*dims[1])+100, 
                     modulus(kk-H+(DIM3)*coords[2], DIM3*dims[2])+100) 
            ) {
          passed = false;
          file << "c " << c[(DIM1+2*H)*(DIM2+2*H)*kk+ii*(DIM2+2*H)+jj] << " != " 
               << triple_t(modulus(ii-H+(DIM1)*coords[0], DIM1*dims[0])+100,
                           modulus(jj-H+(DIM2)*coords[1], DIM2*dims[1])+100, 
                           modulus(kk-H+(DIM3)*coords[2], DIM3*dims[2])+100) 
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
