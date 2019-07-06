/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../../meta.hpp"
#include "./esf.hpp"

namespace gridtools {
    namespace lazy {
        template <class Esf>
        struct esf_param_list {
            GT_STATIC_ASSERT(is_esf_descriptor<Esf>::value, GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT(Esf::location_type::n_colors::value > 0, GT_INTERNAL_ERROR);

            template <class I>
            using nth_param = typename Esf::template esf_function<I::value>::param_list;

            using colors_t = meta::make_indices_c<Esf::location_type::n_colors::value>;
            using param_lists_t = meta::transform<nth_param, colors_t>;

            GT_STATIC_ASSERT(meta::all_are_same<param_lists_t>::value,
                "Multiple Color specializations of the same ESF must contain the same param list");

            using type = meta::first<param_lists_t>;
        };

        template <class Esf, class Args>
        struct esf_replace_args;

        template <template <uint_t> class F, class Grid, class Location, class OldArgs, class NewArgs>
        struct esf_replace_args<esf_descriptor<F, Grid, Location, OldArgs>, NewArgs> {
            using type = esf_descriptor<F, Grid, Location, NewArgs>;
        };
    } // namespace lazy
    GT_META_DELEGATE_TO_LAZY(esf_param_list, class Esf, Esf);
    GT_META_DELEGATE_TO_LAZY(esf_replace_args, (class Esf, class Args), (Esf, Args));

} // namespace gridtools
