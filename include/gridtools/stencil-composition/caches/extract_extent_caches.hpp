/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/copy_into_variadic.hpp"
#include "../../meta.hpp"
#include "../esf_metafunctions.hpp"
#include "../extent.hpp"
#include "./cache_traits.hpp"

namespace gridtools {
    namespace extract_extent_caches_impl_ {
        template <class Arg>
        struct arg_extent_from_esf {

            template <class EsfArg, class Accessor>
            GT_META_DEFINE_ALIAS(
                get_extent, meta::if_, (std::is_same<Arg, EsfArg>, typename Accessor::extent_t, extent<>));

            template <class Esf, class Args = typename Esf::args_t>
            GT_META_DEFINE_ALIAS(apply,
                meta::rename,
                (enclosing_extent,
                    GT_META_CALL(meta::transform,
                        (get_extent, Args, copy_into_variadic<typename esf_param_list<Esf>::type, meta::list<>>))));
        };
    } // namespace extract_extent_caches_impl_

    template <class Arg,
        class Esfs,
        class Extents = GT_META_CALL(
            meta::transform, (extract_extent_caches_impl_::arg_extent_from_esf<Arg>::template apply, Esfs))>
    GT_META_DEFINE_ALIAS(extract_k_extent_for_cache, meta::rename, (enclosing_extent, Extents));

} // namespace gridtools
