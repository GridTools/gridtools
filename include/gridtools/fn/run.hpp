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

#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "../sid/composite.hpp"
#include "./stencil_stage.hpp"

namespace gridtools::fn {
    namespace run_impl_ {
        template <class Sids>
        auto make_composite(Sids &&sids) {
            using keys_t = meta::iseq_to_list<std::make_integer_sequence<int, std::tuple_size_v<Sids>>,
                sid::composite::keys,
                integral_constant>;
            return tuple_util::convert_to<keys_t::template values>(std::forward<Sids>(sids));
        }

        template <class T>
        struct is_stencil_stage : std::false_type {};

        template <class Stencil, class MakeIterator, int Out, int... Ins>
        struct is_stencil_stage<stencil_stage<Stencil, MakeIterator, Out, Ins...>> : std::true_type {};

        template <class... Stages>
        struct is_stencil_stage<merged_stencil_stage<Stages...>> : std::true_type {};

        template <class Backend, class Vertical, class StageSpecs, class Domain, class Sids, class Seeds>
        void run(Backend, Vertical, StageSpecs, Domain const &domain, Sids &&sids, Seeds &&seeds) {
            auto composite = make_composite(std::forward<Sids>(sids));
            tuple_util::for_each(
                [&](auto stage, auto seed) {
                    if constexpr (is_stencil_stage<decltype(stage)>()) {
                        apply_stencil_stage(Backend(), domain, std::move(stage), composite);
                    } else {
                        apply_column_stage(Backend(), domain, std::move(stage), composite, Vertical(), std::move(seed));
                    }
                },
                meta::rename<std::tuple, StageSpecs>(),
                std::forward<Seeds>(seeds));
        }
    } // namespace run_impl_

    using run_impl_::run;
} // namespace gridtools::fn
