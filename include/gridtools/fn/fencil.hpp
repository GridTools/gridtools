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
        template <template <class...> class L, class... Ts, class Refs>
        auto select(L<Ts...>, Refs const &refs) {
            return std::tie(std::get<Ts::value>(refs)...);
        }

        template <class...>
        struct stage {};

        template <auto Stencil, auto Domain, size_t Out, size_t... Ins>
        using make_stage = stage<meta::val<Stencil>,
            meta::val<Domain>,
            std::integral_constant<size_t, Out>,
            meta::list<std::integral_constant<size_t, Ins>...>>;

        template <class Backend, template <class...> class L, class... Stages, class Domain, class Refs>
        void exec_stages(Backend be, L<Stages...>, Domain const &domain, Refs const &refs) {
            (...,
                (fn_apply(be,
                    meta::second<Stages>::value(domain),
                    meta::first<Stages>(),
                    std::get<meta::third<Stages>::value>(refs),
                    select(meta::at_c<Stages, 3>(), refs))));
        }

        template <class Backend, class... Stages>
        constexpr auto fencil = [](auto const &domain, auto &&... args) {
            exec_stages(Backend(), meta::list<Stages...>(), domain, std::tie(args...));
        };
    } // namespace fencil_impl_
    using fencil_impl_::exec_stages;
    using fencil_impl_::fencil;
    using fencil_impl_::make_stage;
    using fencil_impl_::stage;
} // namespace gridtools::fn
