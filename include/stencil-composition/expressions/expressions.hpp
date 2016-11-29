#pragma once

#include "common/defs.hpp"
#include "common/gt_math.hpp"
#include "common/string_c.hpp"

/**@file
   @brief Expression templates definition.
   The expression templates are a method to parse at compile time the mathematical expression given
   by the user, recognizing the structure and building a syntax tree by recursively nesting
   templates.*/

#ifndef CXX11_ENABLED
#error("this file must be included only when c++11 is supported (i.e. ENABLE_CXX11=ON)")
#endif

/** \section expressions Expressions Definition
    @{
*/
#include "expr_base.hpp"
#include "expr_derivative.hpp"
#include "expr_direct_access.hpp"
#include "expr_pow.hpp"
#include "expr_divide.hpp"
#include "expr_minus.hpp"
#include "expr_plus.hpp"
#include "expr_times.hpp"

namespace gridtools {

    namespace expressions {

        template < int Exponent,
            typename FloatType,
            typename boost::enable_if< typename boost::is_arithmetic< FloatType >::type, int >::type = 0 >
        GT_FUNCTION constexpr FloatType pow(FloatType arg1) {
            return gt_pow< Exponent >::apply(arg1);
        }

        /**Expressions defining the interface for specifiyng a given offset for a specified dimension
           \tparam Coordinate: direction in which to apply the offset
           \param offset: the offset to be applied in the Coordinate direction
        */
        template < ushort_t Coordinate >
        GT_FUNCTION constexpr dimension< Coordinate > operator+(dimension< Coordinate > d1, int const &offset) {
            return dimension< Coordinate >(offset);
        }

        template < ushort_t Coordinate >
        GT_FUNCTION constexpr dimension< Coordinate > operator-(dimension< Coordinate > d1, int const &offset) {
            return dimension< Coordinate >(-offset);
        }

    } // namespace expressions

    template < typename Arg1, typename Arg2 >
    struct is_expr< expr_plus< Arg1, Arg2 > > : boost::mpl::true_ {};

    template < typename Arg1, typename Arg2 >
    struct is_expr< expr_minus< Arg1, Arg2 > > : boost::mpl::true_ {};

    template < typename Arg1, typename Arg2 >
    struct is_expr< expr_times< Arg1, Arg2 > > : boost::mpl::true_ {};

    template < typename Arg1, typename Arg2 >
    struct is_expr< expr_divide< Arg1, Arg2 > > : boost::mpl::true_ {};

    template < typename Arg1 >
    struct is_expr< expr_direct_access< Arg1 > > : boost::mpl::true_ {};

    template < typename Arg1 >
    struct is_expr< expr_derivative< Arg1 > > : boost::mpl::true_ {};

    template < typename Arg1, int Exponent >
    struct is_expr< expr_pow< Arg1, Exponent > > : boost::mpl::true_ {};

} // namespace gridtools
/*@}*/
