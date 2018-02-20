/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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
#include "offset_tuple.hpp"

namespace gridtools {

    /** Trick to make nvcc understand that the tuple is a constant expression*/
    template < typename Pair >
    constexpr dimension< Pair::first > get_dim() {
        return dimension< Pair::first >{Pair::second};
    }

    /** @brief tuple of integers mixing runtime and compile time offsets

        it contains a runtime tuple and a compile-time one, when calling get() a lookup is done first on the
        constexpr tuple, if the offset is not found a (constant time) lookup on the runtime tuple is performed.
        The get_constexpr methods looks only in the compile-time tuple, it is static and always returns a constant
       expression.
     */
    template < typename ArgType, typename... Pair >
    struct offset_tuple_mixed {

        GRIDTOOLS_STATIC_ASSERT(is_offset_tuple< ArgType >::value, GT_INTERNAL_ERROR);
        typedef offset_tuple_mixed< ArgType, Pair... > type;
        static const ushort_t n_dimensions = ArgType::n_dimensions;

        typedef ArgType offset_tuple_t;

        offset_tuple_t m_tuple_runtime;

      protected:
        static const constexpr offset_tuple_t s_tuple_constexpr{get_dim< Pair >()...};
        typedef boost::mpl::vector< static_int< n_dimensions - Pair::first >... > coordinates_t;

      public:
        GT_FUNCTION constexpr offset_tuple_mixed() : m_tuple_runtime() {}

#ifndef __CUDACC__
        template < typename... ArgsRuntime,
            typename T = typename boost::enable_if_c< accumulate(logical_and(),
                boost::mpl::or_< boost::is_integral< ArgsRuntime >, is_dimension< ArgsRuntime > >::type::value...) >::
                type >
        GT_FUNCTION constexpr offset_tuple_mixed(ArgsRuntime const &... args)
            : m_tuple_runtime(args...) {}
#else
        template < typename First,
            typename... ArgsRuntime,
            typename T = typename boost::enable_if_c< accumulate(logical_and(),
                boost::mpl::or_< boost::is_integral< First >, is_dimension< First > >::type::value) >::type >
        GT_FUNCTION constexpr offset_tuple_mixed(First const &first_, ArgsRuntime const &... args)
            : m_tuple_runtime(first_, args...) {}
#endif

        template < int_t I, int_t N >
        GT_FUNCTION constexpr offset_tuple_mixed(offset_tuple_mixed< offset_tuple< I, N > > const &other_)
            : m_tuple_runtime(other_.m_tuple_runtime) {}

        template < int_t I, int_t N >
        GT_FUNCTION constexpr offset_tuple_mixed(offset_tuple< I, N > const &arg_)
            : m_tuple_runtime(arg_) {}

        template < typename OtherAcc >
        GT_FUNCTION constexpr offset_tuple_mixed(offset_tuple_mixed< OtherAcc, Pair... > &&other_)
            : m_tuple_runtime(other_.m_tuple_runtime) {}

        template < typename OtherAcc >
        GT_FUNCTION constexpr offset_tuple_mixed(offset_tuple_mixed< OtherAcc, Pair... > const &other_)
            : m_tuple_runtime(other_.m_tuple_runtime) {}

        /**@brief returns the offset at a specific index Idx

           the lookup for the index Idx is done at compile time, i.e. this method returns in constant time
         */
        template < short_t Idx >
        GT_FUNCTION static constexpr int_t get_constexpr() {
#ifndef __CUDACC__
            GRIDTOOLS_STATIC_ASSERT(
                Idx < s_tuple_constexpr.n_dimensions, "the idx must be smaller than the arg dimension");
            GRIDTOOLS_STATIC_ASSERT(Idx >= 0, "the idx must be larger than 0");
            GRIDTOOLS_STATIC_ASSERT(s_tuple_constexpr.template get< Idx >() >= 0,
                "there is a negative offset. If you did this on purpose recompile with the PEDANTIC_DISABLED flag on.");
#endif
            return s_tuple_constexpr.template get< Idx >();
        }

        /**@brief returns the offset at a specific index Idx

           the lookup for the index Idx is done at compile time, i.e. this method returns in constant time
         */
        template < short_t Idx >
        GT_FUNCTION constexpr int_t get() const {
            return boost::is_same< typename boost::mpl::find< coordinates_t, static_int< Idx > >::type,
                       typename boost::mpl::end< coordinates_t >::type >::type::value
                       ? m_tuple_runtime.template get< Idx >()
                       : s_tuple_constexpr.template get< Idx >();
        }
    };

    template < typename ArgType, typename... T >
    struct is_offset_tuple< offset_tuple_mixed< ArgType, T... > > : boost::mpl::true_ {};

    template < typename ArgType, typename... Pair >
    constexpr typename offset_tuple_mixed< ArgType, Pair... >::offset_tuple_t
        offset_tuple_mixed< ArgType, Pair... >::s_tuple_constexpr;

} // namespace gridtools
