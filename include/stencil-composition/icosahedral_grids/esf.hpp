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
#include "../arg.hpp"
#include "../esf_aux.hpp"
#include "../esf_fwd.hpp"
#include "../icosahedral_grids/grid.hpp"
#include "color.hpp"
#include "common/generic_metafunctions/is_sequence_of.hpp"
#include "extent.hpp"
#include "vector_accessor.hpp"

namespace gridtools {

    template < template < uint_t > class Functor,
        typename Grid,
        typename LocationType,
        typename Color,
        typename ArgSequence >
    struct esf_descriptor {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< ArgSequence, is_arg >::value),
            "wrong types for the list of parameter placeholders\n"
            "check the make_stage syntax");
        GRIDTOOLS_STATIC_ASSERT((is_grid_topology< Grid >::value), "Error: wrong grid type");
        GRIDTOOLS_STATIC_ASSERT((is_color_type< Color >::value), "Error: wrong color type");

        template < uint_t C >
        using esf_function = Functor< C >;
        using grid_t = Grid;
        using location_type = LocationType;
        using args_t = ArgSequence;
        using color_t = Color;

        /** Type member with the mapping between placeholder types (as key) to extents in the operator */
        typedef typename impl::make_arg_with_extent_map< args_t, typename esf_function< 0 >::arg_list >::type
            args_with_extents;
    };

    template < template < uint_t > class Functor,
        typename Grid,
        typename LocationType,
        typename Color,
        typename ArgSequence >
    struct is_esf_descriptor< esf_descriptor< Functor, Grid, LocationType, Color, ArgSequence > > : boost::mpl::true_ {
    };

    template < typename T >
    struct esf_get_location_type;

    template < template < uint_t > class Functor,
        typename Grid,
        typename LocationType,
        typename Color,
        typename ArgSequence >
    struct esf_get_location_type< esf_descriptor< Functor, Grid, LocationType, Color, ArgSequence > > {
        typedef LocationType type;
    };

} // namespace gridtools
