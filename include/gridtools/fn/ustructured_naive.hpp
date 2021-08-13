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

#include "../common/hymap.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "../sid/composite.hpp"
#include "../sid/rename_dimensions.hpp"
#include "../stencil/positional.hpp"
#include "backend/naive.hpp"
#include "naive_apply.hpp"
#include "neighbor_iter.hpp"
#include "strided_iter.hpp"
#include "unstructured.hpp"

namespace gridtools::fn {
    namespace unstructured_naive_impl_ {
        struct hor {};
        using pos_t = stencil::positional<hor>;

        template <class H, class T>
        auto replace_horizontal(T const &obj) {
            using keys_t = meta::rename<hymap::keys, meta::replace<get_keys<T>, H, hor>>;
            return tuple_util::convert_to<keys_t::template values>(obj);
        }

    } // namespace unstructured_naive_impl_

    template <class Sizes, class Offsets, class Horizontal, class Stencil, class Outputs, class Inputs>
    void fn_apply(naive,
        unstructured<Sizes, Offsets, Horizontal> const &domain,
        Stencil const &stencil,
        Outputs &&outputs,
        Inputs &&inputs) {
        using namespace unstructured_naive_impl_;
        pos_t pos;
        auto new_outputs = tuple_util::transform(
            sid::rename_dimensions<Horizontal, unstructured_naive_impl_::hor>, std::forward<Outputs>(outputs));
        naive_apply_impl_::naive_apply(replace_horizontal<Horizontal>(domain.sizes),
            replace_horizontal<Horizontal>(domain.offsets),
            stencil,
            new_outputs,
            std::forward<Inputs>(inputs),
            [&](auto out_tags, auto in_tags, auto &&outs, auto &&ins) {
                using keys_t = meta::rename<sid::composite::keys,
                    meta::push_back<meta::concat<decltype(out_tags), decltype(in_tags)>, pos_t>>;
                return tuple_util::convert_to<keys_t::template values>(tuple_util::push_back(
                    tuple_util::concat(std::forward<decltype(outs)>(outs), std::forward<decltype(ins)>(ins)), pos));
            },
            [](auto &&ptr, auto const &strides) {
                return [index = *at_key<pos_t>(ptr), &ptr, &strides](
                           auto tag) { return neighbor_iter(Horizontal(), index, strided_iter(tag, ptr, strides)); };
            });
    }
} // namespace gridtools::fn
