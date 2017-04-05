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
#include <boost/mpl/at.hpp>
#include "../defs.hpp"
#include "binary_ops.hpp"

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
