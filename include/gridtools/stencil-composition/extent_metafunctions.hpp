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
#include "../common/defs.hpp"
#include "../common/host_device.hpp"
#include "../common/generic_metafunctions/gt_integer_sequence.hpp"
#include "../common/generic_metafunctions/variadic_typedef.hpp"
#include "../common/pair.hpp"
#include "extent.hpp"

namespace gridtools {

    namespace impl {
        template < int Idx, typename Pair >
        struct get_component {

            static constexpr int value = (Idx % 2) ? (Pair::first > Pair::second ? Pair::first : Pair::second)
                                                   : (Pair::first < Pair::second ? Pair::first : Pair::second);
        };
    }

    /**
     * Metafunction taking two extents and yielding a extent containing them
     */
    template < typename Extent1, typename Extent2 >
    struct enclosing_extent_full;

    template < int_t... Vals1, int_t... Vals2 >
    struct enclosing_extent_full< extent< Vals1... >, extent< Vals2... > > {
        GRIDTOOLS_STATIC_ASSERT((sizeof...(Vals1) == sizeof...(Vals2)), "Error: size of the two extents need to match");

        using seq = gridtools::apply_gt_integer_sequence<
            typename gridtools::make_gt_integer_sequence< int, sizeof...(Vals1) >::type >;

        using type = typename seq::
            template apply_t< extent, impl::get_component, ipair_type< int_t, Vals1, Vals2 >... >::type;
    };
}
