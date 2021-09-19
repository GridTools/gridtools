/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <tuple>

#include "../meta.hpp"

namespace gridtools::fn {
    namespace fencil_impl_ {
        template <auto Stencil, auto Domain, size_t Out, size_t... Ins>
        using make_stage = meta::list<meta::val<Stencil>,
            meta::val<Domain>,
            std::integral_constant<size_t, Out>,
            meta::list<std::integral_constant<size_t, Ins>...>>;

        template <class Backend, class... Stages>
        constexpr auto fencil = [](auto const &domain, auto &&... args) {
            fn_fencil(Backend(), meta::list<Stages...>(), domain, std::tie(args...));
        };
    } // namespace fencil_impl_
    using fencil_impl_::fencil;
    using fencil_impl_::make_stage;
} // namespace gridtools::fn
