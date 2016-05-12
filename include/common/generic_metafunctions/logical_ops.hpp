#pragma once

namespace gridtools {

    /**@brief operation to be used inside the accumulator*/
    struct logical_and {
        GT_FUNCTION
        constexpr logical_and() {}
        template < typename T >
        GT_FUNCTION constexpr T operator()(const T &x, const T &y) const {
            return x && y;
        }
    };

    /**@brief operation to be used inside the accumulator*/
    struct logical_or {
        GT_FUNCTION
        constexpr logical_or() {}
        template < typename T >
        GT_FUNCTION constexpr T operator()(const T &x, const T &y) const {
            return x || y;
        }
    };

}
