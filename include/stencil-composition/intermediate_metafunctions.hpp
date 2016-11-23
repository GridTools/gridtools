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

#include "intermediate.hpp"

namespace gridtools {

    template < typename T >
    struct is_intermediate : boost::mpl::false_ {};

    template < typename Backend,
        typename MssArray,
        typename DomainType,
        typename Grid,
        typename ConditionalsSet,
        typename ReductionType,
        bool IsStateful,
        uint_t RepeatFunctor >
    struct is_intermediate<
        intermediate< Backend, MssArray, DomainType, Grid, ConditionalsSet, ReductionType, IsStateful, RepeatFunctor > >
        : boost::mpl::true_ {};

    template < typename T >
    struct intermediate_backend;

    template < typename Backend,
        typename MssArray,
        typename DomainType,
        typename Grid,
        typename ConditionalsSet,
        typename ReductionType,
        bool IsStateful,
        uint_t RepeatFunctor >
    struct intermediate_backend< intermediate< Backend,
        MssArray,
        DomainType,
        Grid,
        ConditionalsSet,
        ReductionType,
        IsStateful,
        RepeatFunctor > > {
        typedef Backend type;
    };

    template < typename T >
    struct intermediate_aggregator_type;

    template < typename Backend,
        typename MssArray,
        typename DomainType,
        typename Grid,
        typename ConditionalsSet,
        typename ReductionType,
        bool IsStateful,
        uint_t RepeatFunctor >
    struct intermediate_aggregator_type< intermediate< Backend,
        MssArray,
        DomainType,
        Grid,
        ConditionalsSet,
        ReductionType,
        IsStateful,
        RepeatFunctor > > {
        typedef DomainType type;
    };

    template < typename T >
    struct intermediate_mss_array;

    template < typename Backend,
        typename MssArray,
        typename DomainType,
        typename Grid,
        typename ConditionalsSet,
        typename ReductionType,
        bool IsStateful,
        uint_t RepeatFunctor >
    struct intermediate_mss_array< intermediate< Backend,
        MssArray,
        DomainType,
        Grid,
        ConditionalsSet,
        ReductionType,
        IsStateful,
        RepeatFunctor > > {
        typedef MssArray type;
    };

    template < typename Intermediate >
    struct intermediate_mss_components_array {
        GRIDTOOLS_STATIC_ASSERT((is_intermediate< Intermediate >::value), "Internal Error: wrong type");
        typedef typename Intermediate::mss_components_array_t type;
    };

    template < typename Intermediate >
    struct intermediate_extent_sizes {
        GRIDTOOLS_STATIC_ASSERT((is_intermediate< Intermediate >::value), "Internal Error: wrong type");
        typedef typename Intermediate::extent_sizes_t type;
    };

    template < typename T >
    struct intermediate_layout_type;

    template < typename T >
    struct intermediate_is_stateful;

    template < typename Backend,
        typename MssArray,
        typename DomainType,
        typename Grid,
        typename ConditionalsSet,
        typename ReductionType,
        bool IsStateful,
        uint_t RepeatFunctor >
    struct intermediate_is_stateful< intermediate< Backend,
        MssArray,
        DomainType,
        Grid,
        ConditionalsSet,
        ReductionType,
        IsStateful,
        RepeatFunctor > > {
        typedef boost::mpl::bool_< IsStateful > type;
    };

    template < typename T >
    struct intermediate_mss_local_domains;

    template < typename Intermediate >
    struct intermediate_mss_local_domains {
        GRIDTOOLS_STATIC_ASSERT((is_intermediate< Intermediate >::value), "Internal Error: wrong type");
        typedef typename Intermediate::mss_local_domains_t type;
    };
} // namespace gridtools
