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
#include "../caches/cache_metafunctions.hpp"
#include "../esf_metafunctions.hpp"

namespace gridtools {
    namespace cuda {
        namespace need_sync_impl_ {
            template <class DirtyPlhs>
            struct is_dirty_f {
                template <class Item, class Extent = typename meta::second<Item>::extent_t>
                using apply = bool_constant<meta::st_contains<DirtyPlhs, meta::first<Item>>::value &&
                                            (Extent::iminus::value != 0 || Extent::iplus::value != 0 ||
                                                Extent::jminus::value != 0 || Extent::jplus::value != 0)>;
            };

            template <class Esf, class DirtyPlhs>
            using has_dirty_args = typename meta::any_of<is_dirty_f<DirtyPlhs>::template apply,
                meta::zip<typename Esf::args_t, esf_param_list<Esf>>>::type;

            template <class State,
                class Esf,
                class DirtyPlhs = meta::second<State>,
                class NeedSync = has_dirty_args<Esf, DirtyPlhs>,
                class OutPlhs = esf_get_w_args_per_functor<Esf>,
                class NewDirtys = meta::if_<NeedSync, OutPlhs, meta::dedup<meta::concat<DirtyPlhs, OutPlhs>>>>
            using folding_fun = meta::list<meta::push_back<meta::first<State>, NeedSync>, NewDirtys>;
        } // namespace need_sync_impl_

        template <class Esfs,
            class Caches,
            class InitialState = meta::list<meta::list<>, meta::list<>>,
            class FinalState = meta::lfold<need_sync_impl_::folding_fun, InitialState, Esfs>,
            class FinalDirty = meta::second<FinalState>,
            class NeedSyncFirst = negation<meta::is_empty<ij_cache_args<Caches>>>>
        using need_sync = meta::replace_at_c<meta::first<FinalState>, 0, NeedSyncFirst>;
    } // namespace cuda
} // namespace gridtools
