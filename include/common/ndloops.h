#ifndef _NDLOOPS_H_
#define _NDLOOPS_H_

#include <stdlib.h>
#include <iostream>

namespace GCL {
  namespace utils {
    template <int D>
    struct prod;

    template <>
    struct prod<-1> {
      template <typename ARRAY>
      int operator()(ARRAY const & dimensions) const {
        return 1;
      }
    };


    template <int D>
    struct prod {
      template <typename ARRAY>
      int operator()(ARRAY const & dimensions) const {
        //    std::cout << D << " prod    " << dimensions[D]*prod<D-1>()(dimensions) << "\n";
        return dimensions[D]*prod<D-1>()(dimensions);
      }
    };


    template <int D>
    struct access_to;

    template <>
    struct access_to<1> {
      template <typename ARRAY>
      int operator()(ARRAY const & indices,
                           ARRAY const &) const {
        //std:: cout << "indices[1] " << indices[1] << "\n";
        return indices[0];
      }
    };

    template <int D>
    struct access_to {
      template <typename ARRAY>
      int operator()(ARRAY const & indices,
                           ARRAY const & dimensions) const {
        //std::cout << access_to<D-1>()(indices,dimensions) << " + "
        //          << indices[D-1] << " * "
        //          << prod<D-2>()(dimensions) << "\n";
        return access_to<D-1>()(indices,dimensions) + indices[D-1]* prod<D-2>()(dimensions);
      }
    };


    struct bounds_sizes {
      int imin;
      int imax;
      int sizes;
    };

    struct bounds {
      int imin;
      int imax;
    };


    template <int I, typename F>
    struct access_loop;

    template <typename F>
    struct access_loop<0,F> {
      template <typename arraybounds, typename array>
      void operator()(arraybounds const & ab,
                      array const &sizes,
                      F &f, int idx=0) {
        f(idx);
      }
    };

    template <int I, typename F>
    struct access_loop {
      template <typename arraybounds, typename array>
      void operator()(arraybounds const & ab,
                      array const &sizes,
                      F &f, int idx=0) {
        int midx;
        for (int i=ab[I-1].imin; i<=ab[I-1].imax; ++i) {
          midx = idx+i* prod<I-2>()(sizes);
          access_loop<I-1,F>()(ab, sizes, f, midx);
        }
      }
    };



    template <int I>
    struct loop;

    template <>
    struct loop<0> {
      template <typename F, typename arraybounds, typename array>
      void operator()(arraybounds const &,
                      F const &f, array & tuple) {
        f(tuple);
      }
    };

    template <int I>
    struct loop {
      template <typename F, typename arraybounds, typename array>
      void operator()(arraybounds const & ab,
                      F const &f, array & tuple) {
        for (int i=ab[I-1].imin; i<=ab[I-1].imax; ++i) {
          tuple[I-1]=i;
          loop<I-1>()(ab, f, tuple);
        }
      }
    };


    template <int I, int LB=-1, int UB=1>
    struct neigh_loop;

    template <int LB, int UB>
    struct neigh_loop<0,LB,UB> {
      template <typename F, typename array>
      void operator()(F &f, array & tuple) {
        f(tuple);
      }
    };

    template <int I, int LB, int UB>
    struct neigh_loop {
      template <typename F, typename array>
      void operator()(F &f, array & tuple) {
        for (int i=LB; i<=UB; ++i) {
          tuple[I-1]=i;
          neigh_loop<I-1,LB,UB>()(f, tuple);
        }
      }
    };


  } //namespace utils
} //namespace GCL

#endif
