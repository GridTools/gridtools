#pragma once
/**
@file
TODO Document me!! :-(
*/

#include "../common/layout_map.hpp"
#include "../common/array.hpp"

namespace gridtools {

    template <typename Layout,
              uint_t N0,
              uint_t N1,
              uint_t N2>
    struct decreasing_stride_order {
        typedef layout_map<N0, N1, N2> sizes;

        static const uint_t stride0 = sizes::template at_<Layout::template pos_<1>::value>::value
            * sizes::template at_<Layout::template pos_<2>::value>::value;
        static const uint_t stride1 = sizes::template at_<Layout::template pos_<2>::value>::value;
        static const uint_t stride2 = 1;

        typedef layout_map<stride0, stride1, stride2> straight_strides;
        typedef layout_map<straight_strides::template at_<Layout::template pos_<0>::value>::value,
                           straight_strides::template at_<Layout::template pos_<1>::value>::value,
                           straight_strides::template at_<Layout::template pos_<2>::value>::value> strides;

    };

    template <typename ValueType,
              typename Layout,
              uint_t _N0,
              uint_t _N1,
              uint_t _N2>
    struct small_storage {
        gridtools::array<ValueType, _N0*_N1*_N2> data;

        typedef typename decreasing_stride_order<Layout, _N0, _N1, _N2>::strides strides;

        ValueType& operator()(uint_t i, uint_t j, uint_t k) {
            return data[_index(i,j,k)];
        }

        ValueType const & operator()(uint_t i, uint_t j, uint_t k) const {
            return data[_index(i,j,k)];
        }

        template <uint_t I>
        GT_FUNCTION
        uint_t stride_along() const {
            return decreasing_stride_order<Layout, _N0, _N1, _N2>::straight_strides::template at_<I>::value;
        }

        GT_FUNCTION
        uint_t offset(uint_t i, uint_t j, uint_t k) const {
            return _index(i,j,k);
        }

        GT_FUNCTION
        uint_t _index(uint_t i, uint_t j, uint_t k) const {
            return i*strides::template at_<0>::value + j*strides::template at_<1>::value + k*strides::template at_<2>::value;
        }

    };
} // namespace gridtools
