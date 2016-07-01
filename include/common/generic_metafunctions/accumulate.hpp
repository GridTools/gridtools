#pragma once
#include <boost/mpl/at.hpp>
#include "logical_ops.hpp"

/**@file @brief implementation of a compile-time accumulator and max

   The accumulator allows to perform operations on static const value to
   be passed as template argument. E.g. to pass the sum of all the
   storage dimensions as template argument of \ref
   gridtools::base_storage This files also contains the implementation of
   some constexpr operations which can be used in the accumulator, and of
   a vec_max operator computing the maximum of a compile time sequence of
   integers.
*/
namespace gridtools {

    /**@brief computes the maximum between two numbers*/
    template < typename T1, typename T2 >
    struct max {
        static const int_t value = (T1::value > T2::value) ? T1::value : T2::value;
        typedef static_int< value > type;
    };

    /**@brief computes the maximum in a sequence of numbers*/
    template < typename Vector, uint_t ID >
    struct find_max {
        typedef typename max< typename boost::mpl::at_c< Vector, ID >::type,
            typename find_max< Vector, ID - 1 >::type >::type type;
    };

    /**@brief specialization to stop the recursion*/
    template < typename Vector >
    struct find_max< Vector, 0 > {
        typedef typename boost::mpl::at_c< Vector, 0 >::type type;
    };

    /**@brief defines the maximum (as a static const value and a type) of a sequence of numbers.*/
    template < typename Vector >
    struct vec_max {
        typedef typename find_max< Vector, boost::mpl::size< Vector >::type::value - 1 >::type type;
        static const int_t value = type::value;
    };

    /**@brief operation to be used inside the accumulator*/
    struct multiplies {
        GT_FUNCTION
        constexpr multiplies() {}
        template < typename T1, typename T2 >
        GT_FUNCTION constexpr T1 operator()(const T1 &x, const T2 &y) const {
            return x * y;
        }
    };

    /**@brief operation to be used inside the accumulator*/
    struct add_functor {
        GT_FUNCTION
        constexpr add_functor() {}
        template < class T >
        GT_FUNCTION constexpr T operator()(const T &x, const T &y) const {
            return x + y;
        }
    };

#ifdef CXX11_ENABLED
    /**@brief accumulator recursive implementation*/
    template < typename Operator, typename First, typename... Args >
    GT_FUNCTION static constexpr First accumulate(Operator op, First first, Args... args) {
        return op(first, accumulate(op, args...));
    }

#ifdef __CUDACC__ // no clue why nvcc cannot figure this out (works on a small test)
    /**@brief accumulator recursive implementation*/
    template < typename First, typename... Args >
    GT_FUNCTION static constexpr First accumulate(add_functor op, First first, Args... args) {
        return op(first, accumulate(op, args...));
    }
#endif

    /**@brief specialization to stop the recursion*/
    template < typename Operator, typename First >
    GT_FUNCTION static constexpr First accumulate(Operator op, First first) {
        return first;
    }

#ifdef __CUDACC__
    /**@brief accumulator recursive implementation*/
    template < typename First >
    GT_FUNCTION static constexpr First accumulate(add_functor op, First first) {
        return first;
    }
#endif

#endif

} // namespace gridtools
