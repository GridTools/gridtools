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
#include "../esf.hpp"
#include "../caches/cache_metafunctions.hpp"

/**
@file
@brief descriptor of the Multi Stage Stencil (MSS)
*/
namespace gridtools {

    template < typename T >
    struct is_cache;

    /** @brief Descriptors for  a reduction type of mss
     * @tparam ReductionType basic type of the fields being reduced
     * @tparam BinOp binary operation applied for the reduction
     * @tparam EsfDescrSequence sequence of esf descriptor (should contain only one esf
     *      with the reduction functor)
     */
    template < typename ReductionType, typename BinOp, typename EsfDescrSequence >
    struct reduction_descriptor {
        GRIDTOOLS_STATIC_ASSERT(
            (is_sequence_of< EsfDescrSequence, is_esf_descriptor >::value), "Internal Error: invalid type");
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< EsfDescrSequence >::value == 1), "Internal Error: invalid type");

        typedef ReductionType reduction_type_t;
        typedef EsfDescrSequence esf_sequence_t;
        typedef boost::mpl::vector0<> cache_sequence_t;
        typedef static_bool< true > is_reduction_t;
        typedef BinOp bin_op_t;

      private:
        reduction_type_t m_initial_value;

      public:
        constexpr reduction_descriptor(ReductionType initial_value) : m_initial_value(initial_value) {}
        constexpr reduction_type_t get() const { return m_initial_value; }
    };

    template < typename Reduction >
    struct is_reduction_descriptor : boost::mpl::false_ {};

    template < typename ReductionType, typename BinOp, typename EsfDescrSequence >
    struct is_reduction_descriptor< reduction_descriptor< ReductionType, BinOp, EsfDescrSequence > >
        : boost::mpl::true_ {};

    template < typename T >
    struct mss_descriptor_esf_sequence;
    template < typename T >
    struct mss_descriptor_cache_sequence;
    template < typename T >
    struct mss_descriptor_execution_engine;
    template < typename T >
    struct mss_descriptor_is_reduction;
    template < typename ReductionType, typename BinOp, typename EsfDescrSequence >
    struct mss_descriptor_esf_sequence< reduction_descriptor< ReductionType, BinOp, EsfDescrSequence > > {
        typedef EsfDescrSequence type;
    };

    template < typename ReductionType, typename BinOp, typename EsfDescrSequence >
    struct mss_descriptor_cache_sequence< reduction_descriptor< ReductionType, BinOp, EsfDescrSequence > > {
        typedef boost::mpl::vector0<> type;
    };

    template < typename ReductionType, typename BinOp, typename EsfDescrSequence >
    struct mss_descriptor_execution_engine< reduction_descriptor< ReductionType, BinOp, EsfDescrSequence > > {
        typedef enumtype::execute< enumtype::forward > type;
    };

    template < typename ReductionType, typename BinOp, typename EsfDescrSequence >
    struct mss_descriptor_is_reduction< reduction_descriptor< ReductionType, BinOp, EsfDescrSequence > > {
        typedef static_bool< true > type;
    };

    template < typename Reduction >
    struct reduction_descriptor_type;

    template < typename ReductionType, typename BinOp, typename EsfDescrSequence >
    struct reduction_descriptor_type< reduction_descriptor< ReductionType, BinOp, EsfDescrSequence > > {
        typedef ReductionType type;
    };

} // namespace gridtools
