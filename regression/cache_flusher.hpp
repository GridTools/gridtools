/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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

#include <assert.h>
#include <vector>
#include <random>
#include <functional>

// class used to flush the cache for OpenMP codes
// initialise with (n>=cache size) to flush all cache
class cache_flusher {
#ifdef __CUDACC__
  public:
    cache_flusher(int n){};
    void flush(){};
#else
    std::vector<double> a_;
    std::vector<double> b_;
    std::vector<double> c_;
    int n_;

  public:
    cache_flusher(int n) {
        assert(n > 2);
        n_ = n / 2;
        a_.resize(n_);
        b_.resize(n_);
        c_.resize(n_);
    };
    void flush() {
        double *a = &a_[0];
        double *b = &b_[0];
        double *c = &c_[0];
        std::default_random_engine gen;
        std::uniform_int_distribution<int> distr(0,n_-1);
        auto dice = std::bind(distr, gen);
#pragma omp parallel for
#pragma vector nontemporal(a)
        for (int i = 0; i < n_; i++)
            a[i] = b[dice()] * c[dice()];
    };
#endif
};
