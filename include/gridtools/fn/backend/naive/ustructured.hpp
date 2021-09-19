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

#include <utility>

#include "../../../common/hymap.hpp"
#include "../../../common/tuple_util.hpp"
#include "../../../meta.hpp"
#include "../../../sid/composite.hpp"
#include "../../../sid/rename_dimensions.hpp"
#include "../../../stencil/positional.hpp"
#include "../../neighbor_iter.hpp"
#include "../../strided_iter.hpp"
#include "../../unstructured.hpp"
#include "apply.hpp"
#include "tag.hpp"

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

    template <auto Stencil, class Sizes, class Offsets, class Horizontal, class Output, class Inputs>
    void fn_apply(naive,
        unstructured<Sizes, Offsets, Horizontal> const &domain,
        meta::val<Stencil>,
        Output &output,
        Inputs inputs) {
        using namespace unstructured_naive_impl_;
        pos_t pos;
        auto out = sid::rename_dimensions<Horizontal, unstructured_naive_impl_::hor>(output);
        naive_apply<Stencil>(replace_horizontal<Horizontal>(domain.sizes),
            replace_horizontal<Horizontal>(domain.offsets),
            out,
            std::move(inputs),
            [&]<class OutTag, class InTags, class Out, class Ins>(OutTag, InTags, Out & out, Ins && ins) {
                using keys_t = meta::rename<sid::composite::keys, meta::push_back<InTags, pos_t, OutTag>>;
                return tuple_util::convert_to<keys_t::template values>(
                    tuple_util::push_back(std::forward<Ins>(ins), pos, out));
            },
            [](auto &&ptr, auto const &strides) {
                return [index = *at_key<pos_t>(ptr), &ptr, &strides](
                           auto tag) { return neighbor_iter(Horizontal(), index, strided_iter(tag, ptr, strides)); };
            });
    }
} // namespace gridtools::fn
