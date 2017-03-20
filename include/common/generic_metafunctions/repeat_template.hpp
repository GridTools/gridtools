/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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
/**
   @file

   Metafunction for creating a template class with an arbitrary length template parameter pack.
   NOTE: gt_integer_sequence can be replaced by std::integer_sequence for C++14 and beyond.
*/

#include "gt_integer_sequence.hpp"

namespace gridtools {

    namespace _impl {
        template < typename T1, typename T2 >
        struct expand;

        template < typename UInt, UInt... I1, UInt... I2 >
        struct expand< gt_integer_sequence< UInt, I1... >, gt_integer_sequence< UInt, I2... > >
            : gt_integer_sequence< UInt, I1..., I2... > {};

        template < typename UInt, UInt C, ushort_t N >
        struct expand_to_gt_integer_sequence
            : expand< typename expand_to_gt_integer_sequence< UInt, C, N / 2 >::type,
                  typename expand_to_gt_integer_sequence< UInt, C, N - N / 2 >::type >::type {};

        template < typename UInt, UInt C >
        struct expand_to_gt_integer_sequence< UInt, C, 0 > : gt_integer_sequence< UInt > {};
        template < typename UInt, UInt C >
        struct expand_to_gt_integer_sequence< UInt, C, 1 > : gt_integer_sequence< UInt, C > {};

        template < typename Seq, template < ushort_t... > class Lambda, ushort_t... InitialValues >
        struct expand_recursively;

        template < template < ushort_t... > class Lambda, ushort_t... Ints, ushort_t... InitialValues >
        struct expand_recursively< gt_integer_sequence< ushort_t, Ints... >, Lambda, InitialValues... > {
            typedef Lambda< InitialValues..., Ints... > type;
        };
    }

#if defined(CXX11_ENABLED) && !defined(__CUDACC__)

    namespace _impl {
        template < typename Seq, typename Value, template < Value... > class Lambda, Value... InitialValues >
        struct expand_recursively_v;

        template < typename Value, template < Value... > class Lambda, Value... Ints, Value... InitialValues >
        struct expand_recursively_v< gt_integer_sequence< Value, Ints... >, Value, Lambda, InitialValues... > {
            typedef Lambda< InitialValues..., Ints... > type;
        };
    }
    /**
       @brief Metafunction for creating a template class with an arbitrary length template parameter pack.

       Usage example:
       I have a class template halo< .... >, and I want to fill it by repeating N times the same number H (represented
       as an MPL type).
       The following example
       \verbatim
       repeat_template_v<static_short<5>, static_short<3>, short_t, halo, 4, 7, 8>
       \endverbatim
       will generate the type
       \verbatim
        halo<4, 7, 8, 5, 5, 5>
       \endverbatim
       Optionally a set of initial values to start filling the template class can be passed
     */
    template < typename Constant,
        typename Length,
        typename Value,
        template < Value... T > class Lambda,
        Value... InitialValues >
    struct repeat_template_v {
        typedef typename _impl::expand_recursively_v<
            typename _impl::expand_to_gt_integer_sequence< Value, Constant::value, Length::value >::type,
            Value,
            Lambda,
            InitialValues... >::type type;
    };

    /**
       @brief Metafunction for creating a template class with an arbitrary length template parameter pack.

       Usage example:
       I have a class template halo< .... >, and I want to fill it by repeating N times the same number H
       \verbatim
       repeat_template_v_c<H, N, halo>
       \endverbatim
       Optionally a set of initial values to start filling the template class can be passed
    */
    template < ushort_t Constant,
        ushort_t Length,
        typename Value,
        template < Value... T > class Lambda,
        Value... InitialValues >
    struct repeat_template_v_c {
        typedef typename _impl::expand_recursively_v<
            typename _impl::expand_to_gt_integer_sequence< Value, Constant, Length >::type,
            Value,
            Lambda,
            InitialValues... >::type type;
    };
#endif

    /**
     * similar to repeat_template_v_c with some limitations (used because the general version repeat_template_v_c
     * does not compile with CUDA due to nvcc bug). It assumes that the types of the template parameters of the Lambda
     * function are ushort_t.
     */
    template < ushort_t Constant, ushort_t Length, template < ushort_t... T > class Lambda, ushort_t... InitialValues >
    struct repeat_template_c {
        typedef typename _impl::expand_recursively<
            typename _impl::expand_to_gt_integer_sequence< ushort_t, Constant, Length >::type,
            Lambda,
            InitialValues... >::type type;
    };

} // namespace gridtools
