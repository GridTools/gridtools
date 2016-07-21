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
#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#include <iostream>
#include <common/ndloops.hpp>
#include <common/array.hpp>
#include <sys/time.h>
struct sumup {
  mutable double res;
  int N;
  double *storage;
  sumup(int N, double *st): res(0.0), N(N), storage(st) { }

  template <typename TUPLE>
  void operator()(TUPLE const &tuple) const {
    int idx = tuple[0]+tuple[1]*N+tuple[2]*N*N+tuple[3]*N*N*N;
    res += storage[idx];
  }
};

struct sumup2 {
  mutable double res;
  int N;
  double *storage;
  sumup2(int N, double *st): res(0.0), N(N), storage(st) { }

  void operator()(int idx) const {
    // std::cout << ++i << " " << idx << "\n";
    res += storage[idx];
  }
};

struct print_tuple {
  template <typename TUPLE>
  void operator()(TUPLE const &tuple) const {
    std::cout << "(";
    for (unsigned int i=0; i<tuple.size()-1; ++i) {
      std::cout << tuple[i] << ", ";
    }
    std::cout << tuple[tuple.size()-1] << ") x\n";
  }
};

struct print_int {
  void operator()(int idx) const {
    std::cout << "(" << idx << ")\n";
  }
};


int main(int argc, char** argv) {

  gridtools::array<int, 4> indices; /*= {3, 4, 3, 2}; // enabled in C++0x */
  indices[0] = 3;
  indices[1] = 4;
  indices[2] = 3;
  indices[3] = 2;
  gridtools::array<int, 4> dimensions; /*= {5, 5, 5, 5};  // enabled in C++0x */
  dimensions[0] = 5;
  dimensions[1] = 5;
  dimensions[2] = 5;
  dimensions[3] = 5;
  std::cout << gridtools::access_to<4>()(indices, dimensions) << "\n";

  int N=atoi(argv[1]);

  gridtools::array<gridtools::bounds,4> ab;
//   ab[0].imin=2;
//   ab[0].imax=dimensions[0]-1;
//   ab[1].imin=3;
//   ab[1].imax=dimensions[1]-3;
//   ab[2].imin=10;
//   ab[2].imax=dimensions[2]-2;
//   ab[3].imin=0;
//   ab[3].imax=dimensions[3]-1;
  ab[0].imin=0;
  ab[0].imax=N-1;
  ab[1].imin=0;
  ab[1].imax=N-1;
  ab[2].imin=0;
  ab[2].imax=N-1;
  ab[3].imin=0;
  ab[3].imax=N-1;

  print_int tmp;
  gridtools::access_loop<4,print_int>()(ab,dimensions,tmp);


  struct timeval start_tv;
  struct timeval stop_tv;
  double time;

  std::cout << "\n\n\n\n";
  gridtools::array<int,4> tuple;
  print_tuple tmp2;
  gridtools::loop<4>()(ab,tmp2,tuple);


  double *storage = new double[N*N*N*N];

  for (int i=0; i<N; ++i)
    for (int j=0; j<N; ++j)
      for (int k=0; k<N; ++k)
        for (int l=0; l<N; ++l) {
          int idx = l+k*N+j*N*N+i*N*N*N;
          storage[idx] = i+j+k+l;
        }

  ab[0].imin=0;
  ab[0].imax=N-1;
  ab[1].imin=0;
  ab[1].imax=N-1;
  ab[2].imin=0;
  ab[2].imax=N-1;
  ab[3].imin=0;
  ab[3].imax=N-1;

  dimensions[0]=N;
  dimensions[1]=N;
  dimensions[2]=N;
  dimensions[3]=N;

  std::cout << "start regular\n";

  gettimeofday(&start_tv, NULL);
  double res = 0;
  for (int i=0; i<N; ++i)
    for (int j=0; j<N; ++j)
      for (int k=0; k<N; ++k)
        for (int l=0; l<N; ++l) {
          int idx = l+k*N+j*N*N+i*N*N*N;
          res += storage[idx];
        }
  gettimeofday(&stop_tv, NULL);

  time = (((double)stop_tv.tv_sec+1/1000000.0*(double)stop_tv.tv_usec)
          - ((double)start_tv.tv_sec+1/1000000.0*(double)start_tv.tv_usec)) * 1000.0;

  std::cout << "result " << res << " time " << time << "\n";

  for (int i=0; i<N; ++i)
    for (int j=0; j<N; ++j)
      for (int k=0; k<N; ++k)
        for (int l=0; l<N; ++l) {
          int idx = l+k*N+j*N*N+i*N*N*N;
          storage[idx] = (i+j+k+l)/10.;
        }

  std::cout << "start loop\n";

  sumup summ(N, storage);
  gettimeofday(&start_tv, NULL);
  gridtools::loop<4>()(ab,summ, tuple);
  gettimeofday(&stop_tv, NULL);

  time = (((double)stop_tv.tv_sec+1/1000000.0*(double)stop_tv.tv_usec)
          - ((double)start_tv.tv_sec+1/1000000.0*(double)start_tv.tv_usec)) * 1000.0;

  std::cout << "result " << summ.res << " time " << time << "\n";

  for (int i=0; i<N; ++i)
    for (int j=0; j<N; ++j)
      for (int k=0; k<N; ++k)
        for (int l=0; l<N; ++l) {
          int idx = l+k*N+j*N*N+i*N*N*N;
          storage[idx] = (i+j+k+l)/100.;
        }

  std::cout << "start loop with access function\n";

  sumup2 summ2(N, storage);
  gettimeofday(&start_tv, NULL);
  gridtools::access_loop<4,sumup2>()(ab,dimensions,summ2);
  gettimeofday(&stop_tv, NULL);

  time = (((double)stop_tv.tv_sec+1/1000000.0*(double)stop_tv.tv_usec)
          - ((double)start_tv.tv_sec+1/1000000.0*(double)start_tv.tv_usec)) * 1000.0;

  std::cout << "result " << summ2.res << " time " << time << "\n";

  return 0;
}
