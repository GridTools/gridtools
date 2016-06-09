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
        bool IsStateful >
    struct is_intermediate<
        intermediate< Backend, MssArray, DomainType, Grid, ConditionalsSet, ReductionType, IsStateful > >
        : boost::mpl::true_ {};

    template < typename T >
    struct intermediate_backend;

    template < typename Backend,
        typename MssArray,
        typename DomainType,
        typename Grid,
        typename ConditionalsSet,
        typename ReductionType,
        bool IsStateful >
    struct intermediate_backend<
        intermediate< Backend, MssArray, DomainType, Grid, ConditionalsSet, ReductionType, IsStateful > > {
        typedef Backend type;
    };

    template < typename T >
    struct intermediate_domain_type;

    template < typename Backend,
        typename MssArray,
        typename DomainType,
        typename Grid,
        typename ConditionalsSet,
        typename ReductionType,
        bool IsStateful >
    struct intermediate_domain_type<
        intermediate< Backend, MssArray, DomainType, Grid, ConditionalsSet, ReductionType, IsStateful > > {
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
        bool IsStateful >
    struct intermediate_mss_array<
        intermediate< Backend, MssArray, DomainType, Grid, ConditionalsSet, ReductionType, IsStateful > > {
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
        bool IsStateful >
    struct intermediate_is_stateful<
        intermediate< Backend, MssArray, DomainType, Grid, ConditionalsSet, ReductionType, IsStateful > > {
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
