
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
#include <utils/boollist.h>
#include <utils/layout_map.h>
#include <stdlib.h>

#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/comparison/not_equal.hpp>



#define MACRO_IMPL(z, n, N)                     \
  AA.at( n ) BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(n,N))                                      

inline int modulus(int __i, int __j) {
  return (((((__i%__j)<0)?(__j+__i%__j):(__i%__j))));
}

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

#define N 9
#define GCL_LEFT AA.at ## BOOST_PP_LPAREN()
#define GCL_RIGHT BOOST_PP_RPAREN()
 
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  GCL::GCL_Init(argc, argv);

  int DIM1 = atoi(argv[1]);
  int DIM2 = atoi(argv[2]);
  int DIM3 = atoi(argv[3]);
  int H = atoi(argv[4]); // halo

  if (DIM1 < 4*H)
    exit(1);
  if (DIM2 < 4*H)
    exit(1);
  if (DIM3 < 4*H)
    exit(1);

  int pid;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  std::vector<triple_t*> AA(N);
 
  for (int i=0; i<N; ++i)
    AA[i] = new triple_t[DIM1*(DIM2+pid)*(DIM3+2*pid+1)];

  std::cout << pid << " " << nprocs << " mem: " << DIM1*(DIM2+pid)*(DIM3+2*pid+1)*N*sizeof(triple_t) << "\n";

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
  int period[3] = {1, 1, 1};

  file << "@" << pid << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << " - " << dims[2] << "\n";
 
  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, period, false, &CartComm);

  typedef GCL::gcl_utils::boollist<3> cyc;
  typedef GCL::MPI_3D_process_grid_t<cyc> grid_type;

  cyc periodicity(true, true, true);

  int pi, pj, pk;
  int PI, PJ, PK;

  typedef GCL::halo_exchange_dynamic_ut
    <GCL::layout_map<0,1,2>,
    GCL::layout_map<0,1,2>,
    triple_t,3> HD_TYPE;

#ifdef SPLIT_PHASE
  std::vector<HD_TYPE* > hd(N);

  for (int i=0; i<N; ++i) {
    hd[i]=new HD_TYPE(periodicity, CartComm);

    hd[i]->add_halo<2>(H, H, H, DIM3-H-1, DIM3+2*pid+1);
    hd[i]->add_halo<1>(H, H, H, DIM2-H-1, DIM2+pid);
    hd[i]->add_halo<0>(H, H, H, DIM1-H-1, DIM1);

    hd[i]->setup(1);
  }

  hd[0]->pattern().proc_grid().coords(pk, pj, pi);
  hd[0]->pattern().proc_grid().dims(PK, PJ, PI);

#else
  HD_TYPE hd(periodicity, CartComm);

  hd.add_halo<2>(H, H, H, DIM3-H-1, DIM3+2*pid+1);
  hd.add_halo<1>(H, H, H, DIM2-H-1, DIM2+pid);
  hd.add_halo<0>(H, H, H, DIM1-H-1, DIM1);

  hd.setup(N);

  hd.pattern().proc_grid().coords(pk, pj, pi);
  hd.pattern().proc_grid().dims(PK, PJ, PI);

#endif


  file << "COORDINATES " << pi << ", " << pj << ", " << pk << std::endl;

  // FIELD I has halo I
  // filling data
  for (int l=0; l<N; ++l) {
    for (int i=H; i<DIM3-H; i++)
      for (int j=H; j<DIM2-H; j++)
        for (int k=H; k<DIM1-H; k++)
          AA[l][GCL::access(k,j,i,DIM1,DIM2+pid,DIM3+2*pid+1)] = triple_t(i-H+(DIM3-2*H)*pi+l, 
                                                                          j-H+(DIM2-2*H)*pj+l, 
                                                                          k-H+(DIM1-2*H)*pk+l);        
  }

  int maxi, maxj, maxk;
  maxi = (DIM3-2*H)*PI; // this is to compute modulo in cyclic boundary communications
  maxj = (DIM2-2*H)*PJ; // this is to compute modulo in cyclic boundary communications
  maxk = (DIM1-2*H)*PK; // this is to compute modulo in cyclic boundary communications

  std::cout << "Overall problem size: " 
            << maxi << " x " 
            << maxj << " x " 
            << maxk << "\n"; 

#ifdef VECTOR_INTERFACE
#ifdef SPLIT_PHASE
  file << "testing vector interface - split-phase\n";
  for (int i=0; i<N; ++i) {
    hd[i]->pack(std::vector<triple_t*>(1,AA[i]));
    hd[i]->start_exchange();
  }

  for (int i=0; i<N; ++i) {
    hd[i]->wait();
    hd[i]->unpack(std::vector<triple_t*>(1,AA[i]));
  }
#else
  file << "testing vector interface\n";
  hd.pack(AA);
  
  hd.exchange();
  
  hd.unpack(AA);
#endif
#else
#ifdef SPLIT_PHASE
  file << "testing variable argument interface - split-phase\n";
  for (int i=0; i<N; ++i) {
    hd[i]->pack(AA[i]);
    hd[i]->start_exchange();
  }

  for (int i=0; i<N; ++i) {
    hd[i]->wait();
    hd[i]->unpack(AA[i]);
  }
#else
  file << "testing variable argument list interface\n";
  hd.pack(     
          BOOST_PP_REPEAT(N, MACRO_IMPL, BOOST_PP_DEC(N))
               );
  
  
  hd.exchange();
  
  hd.unpack(     
          BOOST_PP_REPEAT(N, MACRO_IMPL, BOOST_PP_DEC(N))
               );
#endif
#endif

  //  hd.unpack(BOOST_PP_ENUM_BINARY_PARAMS(BOOST_PP_INC(N), AA.at( , ) BOOST_PP_INTERCEPT));

  bool passed=true;

  for (int l=0; l<N; ++l) {
    for (int i=((pi>0)?0:((periodicity.value2)?0:H)); i<((pi<PI-1)?DIM3:((periodicity.value2)?DIM3:DIM3-H)); i++)
      for (int j=((pj>0)?0:((periodicity.value1)?0:H)); j<((pj<PJ-1)?DIM2:((periodicity.value1)?DIM2:DIM2-H)); j++)
        for (int k=((pk>0)?0:((periodicity.value0)?0:H)); k<((pk<PK-1)?DIM1:((periodicity.value0)?DIM1:DIM1-H)); k++)
          if (AA[l][GCL::access(k,j,i,DIM1,DIM2+pid,DIM3+2*pid+1)] != triple_t(modulus(i-H+(DIM3-2*H)*pi,maxi)+l, 
                                                                               modulus(j-H+(DIM2-2*H)*pj,maxj)+l, 
                                                                               modulus(k-H+(DIM1-2*H)*pk,maxk)+l))
            {
              file << l << " " 
                   << i << " " 
                   << j << " " 
                   << k << " " 
                   << AA[l][GCL::access(k,j,i,DIM1,DIM2+pid,DIM3+2*pid+1)] << " "
                   << triple_t(modulus(i-H+(DIM3-2*H)*pi,maxi)+l, 
                               modulus(j-H+(DIM2-2*H)*pj,maxj)+l, 
                               modulus(k-H+(DIM1-2*H)*pk,maxk)+l)
                   << " -- modulo " << maxi << " " << maxj << " " << maxk 
                   << std::endl;
              passed=false;
            }
  }
  
  if (passed)
    file << "RESULT: PASSED\n";
  else
    file << "RESULT: FAILED\n";

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
