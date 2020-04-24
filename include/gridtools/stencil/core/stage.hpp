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

namespace gridtools {
    namespace {
        namespace stage_impl_ {
            template <class>
            struct meta_stage;

            template <template <class...> class L, class... Ts>
            struct meta_stage<L<Ts...>> : decltype(get_stage(std::declval<Ts>()...)) {};
        } // namespace stage_impl_
        template <class Functor, class PlhMap>
        using stage = typename stage_impl_::meta_stage<typename Functor::param_list>::template apply<Functor, PlhMap>;
    } // namespace
} // namespace gridtools
