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
#include "backend/naive.hpp"
#include "cartesian.hpp"
#include "naive_apply.hpp"
#include "strided_iter.hpp"

namespace gridtools::fn {
    template <class Sizes, class Offsets, class Stencil, class Outputs, class Inputs>
    void fn_apply(
        naive, cartesian<Sizes, Offsets> const &domain, Stencil const &stencil, Outputs &&outputs, Inputs &&inputs) {
        naive_apply(domain.sizes,
            domain.offsets,
            stencil,
            std::forward<Outputs>(outputs),
            std::forward<Inputs>(inputs),
            [](auto out_tags, auto in_tags, auto &&outs, auto &&ins) {
                using keys_t = meta::rename<sid::composite::keys, meta::concat<decltype(out_tags), decltype(in_tags)>>;
                return tuple_util::convert_to<keys_t::template values>(
                    tuple_util::concat(std::forward<decltype(outs)>(outs), std::forward<decltype(ins)>(ins)));
            },
            [](auto &&ptr, auto const &strides) { return [&](auto tag) { return strided_iter(tag, ptr, strides); }; });
    }
} // namespace gridtools::fn
