/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#pragma once
#include "offset_tuple.hpp"

#ifdef CUDA8
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

        GRIDTOOLS_STATIC_ASSERT(is_offset_tuple< ArgType >::value, "wrong type");
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

        template < typename OffsetTuple, int_t I, int_t N >
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
            GRIDTOOLS_STATIC_ASSERT(Idx < s_tuple_constexpr.n_dimensions, "the idx must be smaller than the arg dimension");
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
#endif // CUDA8
