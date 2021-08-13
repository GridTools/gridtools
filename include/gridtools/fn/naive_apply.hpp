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
#include <type_traits>
#include <utility>

#include "../common/hymap.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "../sid/concept.hpp"
#include "../sid/loop.hpp"
#include "../sid/multi_shift.hpp"

namespace gridtools::fn {
    namespace naive_apply_impl_ {
        template <class T>
        struct out_tag : T {};

        template <class T>
        struct in_tag : T {};

        template <class Src>
        auto to_tuple(Src &&src) {
            if constexpr (meta::is_instantiation_of<std::tuple, std::decay_t<Src>>())
                return std::forward<Src>(src);
            else
                return std::tuple(std::forward<Src>(src));
        }

        template <class Sizes>
        auto make_loops(Sizes const &sizes) {
            return tuple_util::fold(
                [&](auto outer, auto dim) {
                    using dim_t = decltype(dim);
                    return [outer = std::move(outer), inner = sid::make_loop<dim_t>(at_key<dim_t>(sizes))](
                               auto &&... args) { return outer(inner(std::forward<decltype(args)>(args)...)); };
                },
                std::identity(),
                meta::rename<std::tuple, get_keys<Sizes>>());
        }

        template <class OutTags, class InTags, class Stencil, class MakeIterator>
        auto make_body(Stencil const &stencil, MakeIterator &&make_iterator) {
            return [&](auto const &ptr, auto const &strides) {
                auto srcs = to_tuple(std::apply(stencil, tuple_util::transform(make_iterator(ptr, strides), InTags())));
                for_each<OutTags>([&ptr, &srcs](auto tag) {
                    using tag_t = decltype(tag);
                    *at_key<tag_t>(ptr) = std::get<tag_t::value>(srcs);
                });
            };
        }

        template <class Sizes,
            class Offsets,
            class Stencil,
            class Outputs,
            class Inputs,
            class MakeComposite,
            class MakeIterator>
        void naive_apply(Sizes const &sizes,
            Offsets const &offsets,
            Stencil const &stencil,
            Outputs &&outputs,
            Inputs &&inputs,
            MakeComposite &&make_composite,
            MakeIterator &&make_iterator) {
            using out_tags_t = meta::transform<out_tag, meta::make_indices<tuple_util::size<Outputs>, std::tuple>>;
            using in_tags_t = meta::transform<in_tag, meta::make_indices<tuple_util::size<Inputs>, std::tuple>>;
            auto composite = std::forward<MakeComposite>(make_composite)(out_tags_t(), in_tags_t(), outputs, inputs);
            auto strides = sid::get_strides(composite);
            auto ptr = sid::get_origin(composite)();
            sid::multi_shift(ptr, strides, offsets);
            make_loops(sizes)(make_body<out_tags_t, in_tags_t>(stencil, std::forward<MakeIterator>(make_iterator)))(
                ptr, strides);
        }
    } // namespace naive_apply_impl_
    using naive_apply_impl_::naive_apply;
} // namespace gridtools::fn
