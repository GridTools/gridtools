#pragma once

#include "../common/layout_map.h"
#include "../common/array.h"

namespace gridtools {

    template <typename Layout,
              int N0,
              int N1,
              int N2>
    struct decreasing_stride_order {
        typedef layout_map<N0, N1, N2> sizes;

        static const int stride0 = sizes::template at_<Layout::template pos_<1>::value>::value
            * sizes::template at_<Layout::template pos_<2>::value>::value;
        static const int stride1 = sizes::template at_<Layout::template pos_<2>::value>::value;
        static const int stride2 = 1;

        typedef layout_map<stride0, stride1, stride2> straight_strides;
        typedef layout_map<straight_strides::template at_<Layout::template pos_<0>::value>::value,
                           straight_strides::template at_<Layout::template pos_<1>::value>::value,
                           straight_strides::template at_<Layout::template pos_<2>::value>::value> strides;

    };

    template <typename ValueType,
              typename Layout,
              int _N0,
              int _N1,
              int _N2>
    struct small_storage {
        gridtools::array<ValueType, _N0*_N1*_N2> data;

        typedef typename decreasing_stride_order<Layout, _N0, _N1, _N2>::strides strides;

        ValueType& operator()(int i, int j, int k) {
            return data[_index(i,j,k)];
        }

        ValueType const & operator()(int i, int j, int k) const {
            return data[_index(i,j,k)];
        }

        template <int I>
        GT_FUNCTION
        int stride_along() const {
            return decreasing_stride_order<Layout, _N0, _N1, _N2>::straight_strides::template at_<I>::value;
        }

        GT_FUNCTION
        int offset(int i, int j, int k) const {
            return _index(i,j,k);
        }

        GT_FUNCTION
        int _index(int i, int j, int k) const {
            return i*strides::template at_<0>::value + j*strides::template at_<1>::value + k*strides::template at_<2>::value;
        }

    };
} // namespace gridtools 
