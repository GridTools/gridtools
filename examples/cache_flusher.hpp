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
