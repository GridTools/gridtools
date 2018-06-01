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

#include "../../common/numerics.hpp"
#include "../compute_extents_metafunctions.hpp"
#include "../storage_wrapper.hpp"
#include "../tile.hpp"
#include "grid_traits_backend_fwd.hpp"

#include <boost/mpl/fold.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/max_element.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/quote.hpp>
#include <boost/mpl/vector.hpp>

#ifdef __CUDACC__
#include "backend_cuda/grid_traits_cuda.hpp"
#endif
#include "backend_mic/grid_traits_mic.hpp"
#include "backend_host/grid_traits_host.hpp"

namespace gridtools {

    template <>
    struct grid_traits_from_id< enumtype::structured > {
        // index positions of the different dimensions in the layout map (convention)
        typedef static_uint< 0 > dim_i_t;
        typedef static_uint< 1 > dim_j_t;
        typedef static_uint< 2 > dim_k_t;

        template < class >
        struct make_tmp_data_store_f;

        template < uint_t I, class DataStore >
        struct make_tmp_data_store_f< arg< I, DataStore, enumtype::default_location_type, true > > {
            template < class Size3D >
            DataStore operator()(Size3D const &size) const {
                return {typename DataStore::storage_info_t{size[0], size[1], size[2]}};
            }
        };

        struct select_mss_compute_extent_sizes {
            template < typename PlaceholdersMap, typename Mss >
            struct apply {
                typedef typename compute_extents_of< PlaceholdersMap >::template for_mss< Mss >::type type;
            };
        };

        template < typename Placeholders >
        struct select_init_map_of_extents {
            typedef typename init_map_of_extents< Placeholders >::type type;
        };

        template < enumtype::platform BackendId >
        struct with_arch {
            typedef strgrid::grid_traits_arch< BackendId > type;
        };
    };
}
