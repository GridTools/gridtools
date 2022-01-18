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

#include "gridtools/meta/debug.hpp"
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta.hpp>
#include <gridtools/sid/composite.hpp>

namespace gridtools::fn {
    namespace run_impl_ {
        template <class Sids>
        auto make_composite(Sids &&sids) {
            using keys_t = meta::iseq_to_list<std::make_integer_sequence<int, std::tuple_size_v<Sids>>,
                sid::composite::keys,
                integral_constant>;
            return tuple_util::convert_to<keys_t::template values>(std::forward<Sids>(sids));
        }

        template <class Backend, class StageSpecs, class Domain, class Sids>
        void run(Backend, StageSpecs, Domain const &domain, Sids &&sids) {
            auto composite = make_composite(std::forward<Sids>(sids));
            tuple_util::for_each(
                [&](auto stage) { apply_stencil_stage(Backend(), domain, std::move(stage), composite); },
                meta::rename<std::tuple, StageSpecs>());
        }
    } // namespace run_impl_

    using run_impl_::run;
} // namespace gridtools::fn
