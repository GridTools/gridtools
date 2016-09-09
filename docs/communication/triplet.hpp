/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once
#define USE_DOUBLE false

#ifndef USE_DOUBLE
#define USE_DOUBLE true
#else
#pragma message("USE_DOUBLE already defined, good!")
#endif

template <typename T, typename lmap>
struct array {
  T *ptr;
  int n,m,l;

  array(T* _p, int _n, int _m, int _l)
    : ptr(_p)
    , n(lmap::template find<0>(_n,_m,_l))
    , m(lmap::template find<1>(_n,_m,_l))
    , l(lmap::template find<2>(_n,_m,_l))  
  {}

  T &operator()(int i, int j, int k) {
    // a[(DIM1+2*H)*(DIM2+2*H)*kk+ii*(DIM2+2*H)+jj]
    return ptr[l*m*lmap::template find<0>(i,j,k)+
               l*lmap::template find<1>(i,j,k)+
               lmap::template find<2>(i,j,k)];
  }

  T const &operator()(int i, int j, int k) const {
    return ptr[l*m*lmap::template find<0>(i,j,k)+
               l*lmap::template find<1>(i,j,k)+
               lmap::template find<2>(i,j,k)];
  }

  operator void*() const {return reinterpret_cast<void*>(ptr);}
  operator T*() const {return ptr;}
};

/** \file Example of use of halo_exchange pattern for regular
    grids. The comments in the code aim at highlight the process of
    instantiating and running a halo exchange pattern.
*/

inline int modulus(int __i, int __j) {
  return (((((__i%__j)<0)?(__j+__i%__j):(__i%__j))));
}



/* Just and utility to print values
 */
template <typename array_t>
void printbuff(std::ostream &file, array_t const & a, int d1, int d2, int d3) {
  if (d1<=10 && d2<=10 && d3<=6) {
    file << "------------\n";
    for (int kk=0; kk<d3; ++kk) {
      for (int jj=0; jj<d2; ++jj) {
        file << "|";
        for (int ii=0; ii<d1; ++ii) {
          file << a(ii,jj,kk);
        }
        file << "|\n";
      }
      file << "\n\n";
    }
    file << "------------\n\n";
  }
}

template <bool use_double, typename VT=double>
struct triple_t;

template <typename VT>
struct triple_t </*use_double=*/false, VT>{

  typedef triple_t<false, VT> data_type;

  VT _x,_y,_z;
  __host__ __device__ triple_t(VT a, VT b, VT c): _x(a), _y(b), _z(c) {}
  __host__ __device__ triple_t(): _x(-1), _y(-1), _z(-1) {}

  __host__ __device__ triple_t(triple_t<false,VT> const & t)
    : _x(t._x)
    , _y(t._y)
    , _z(t._z)
  {}

  triple_t<false,VT> floor() {
    VT m = std::min(_x, std::min(_y,_z));
    
    return (m==-1)?triple_t<false,VT>(m,m,m):*this;
  }

  VT x() const {return _x;}
  VT y() const {return _y;}
  VT z() const {return _z;}
};

template <typename VT>
struct triple_t</*use_double=*/true, VT> {

  typedef double data_type;

  double value;

  __host__ __device__ triple_t(int a, int b, int c): value(static_cast<long long int>(a)*100000000+static_cast<long long int>(b)*10000+static_cast<long long int>(c)) {}

  __host__ __device__ triple_t(): value(999999999999) {}

  __host__ __device__ triple_t(triple_t<true,VT> const & t)
    : value(t.value)
  {}

  triple_t<true, VT> floor() {
    if (x() == 9999 || y() == 9999 || z() == 9999) {
      return triple_t<true,VT>();
    } else {
      return *this;
    }
  }

  int x() const {
    long long int cast = static_cast<long long int>(value);
    return static_cast<int>((cast/100000000)%10000);
  }

  int y() const {
    long long int cast = static_cast<long long int>(value);
    return static_cast<int>((cast/10000)%10000);
  }

  template <typename T>
  int y(T& file) const {
    long long int cast = static_cast<long long int>(value);
    file << "$#$@%! " << cast << " " << static_cast<int>((cast/10000)%10000) << std::endl;
    return static_cast<int>((cast/10000)%10000);
  }

  int z() const {
    long long int cast = static_cast<long long int>(value);
    return static_cast<int>((cast)%10000);
  }

};

template <bool V, typename T>
triple_t<V, T> operator*(int a, triple_t<V, T> const& b) {
  return triple_t<V, T>(a*b.x(), a*b.y(), a*b.z());
}

template <bool V, typename T>
triple_t<V, T> operator+(int a, triple_t<V, T> const& b) {
  return triple_t<V, T>(a+b.x(), a+b.y(), a+b.z());
}

template <bool V, typename T>
triple_t<V, T> operator+(triple_t<V, T> const& a, triple_t<V, T> const& b) {
  return triple_t<V, T>(a.x()+b.x(), a.y()+b.y(), a.z()+b.z());
}

template <bool V, typename T>
std::ostream& operator<<(std::ostream &s, triple_t<V, T> const & t) { 
  return s << " (" 
           << t.x() << ", "
           << t.y() << ", "
           << t.z() << ") ";
}

template <bool V, typename T>
bool operator==(triple_t<V, T> const & a, triple_t<V, T> const & b) {
  return (a.x() == b.x() && 
          a.y() == b.y() &&
          a.z() == b.z());
}

template <bool V, typename T>
bool operator!=(triple_t<V, T> const & a, triple_t<V, T> const & b) {
  return !(a==b);
}

