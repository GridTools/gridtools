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
