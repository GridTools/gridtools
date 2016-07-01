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
#pragma once

#include <vector>
#include <assert.h>

// class used to flush the cache for OpenMP codes
// initialise with (n>=cache size) to flush all cache
class cache_flusher {
#ifdef __CUDACC__
  public:
    cache_flusher(int n){};
    void flush(){};
#else
    std::vector< double > a_;
    std::vector< double > b_;
    std::vector< double > c_;
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
        int i;
#pragma omp parallel for private(i)
        for (i = 0; i < n_; i++)
            a[i] = b[i] * c[i];
    };
#endif
};
