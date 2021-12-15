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

#include "../../../common/hymap.hpp"
#include "../../../common/tuple_util.hpp"
#include "../../../meta.hpp"
#include "../../../sid/concept.hpp"
#include "../../../sid/loop.hpp"
#include "../../../sid/multi_shift.hpp"

namespace gridtools::fn {
    namespace naive_apply_impl_ {
        struct out_tag {};

        template <class T>
        struct in_tag : T {};

        template <class Sizes>
        auto make_loops(Sizes const &sizes) {
            return tuple_util::fold(
                [&]<class Outer, class Dim>(Outer outer, Dim) {
                    return [ outer = std::move(outer), inner = sid::make_loop<Dim>(at_key<Dim>(sizes)) ]<class... Args>(
                        Args && ...args) {
                        return outer(inner(std::forward<Args>(args)...));
                    };
                },
                std::identity(),
                meta::rename<std::tuple, get_keys<Sizes>>());
        }

        template <auto Stencil,
            class Sizes,
            class Offsets,
            class Output,
            class Inputs,
            class MakeComposite,
            class MakeIterator>
        void naive_apply(Sizes const &sizes,
            Offsets const &offsets,
            Output &output,
            Inputs &&inputs,
            MakeComposite &&make_composite,
            MakeIterator &&make_iterator) {
            using in_tags_t = meta::transform<in_tag, meta::make_indices<tuple_util::size<Inputs>, std::tuple>>;
            auto composite = std::forward<MakeComposite>(make_composite)(out_tag(), in_tags_t(), output, inputs);
            auto strides = sid::get_strides(composite);
            auto ptr = sid::get_origin(composite)();
            sid::multi_shift(ptr, strides, offsets);
            make_loops(sizes)([&](auto const &ptr, auto const &strides) {
                *at_key<out_tag>(ptr) =
                    std::apply(Stencil, tuple_util::transform(make_iterator(ptr, strides), in_tags_t()));
            })(ptr, strides);
        }

        template <class Key>
        struct at_generator {
            auto const &operator()(auto const &m) const { return at_key<Key>(m); }
        };

        template <class Key, class Map>
        auto remove_key(Map const &m) {
            using res_t = hymap::from_meta_map<meta::mp_remove<hymap::to_meta_map<Map>, Key>>;
            using generators_t = meta::transform<at_generator, get_keys<res_t>>;
            return tuple_util::generate<generators_t, res_t>(m);
        }

        template <class Vertical,
            class IsBackward,
            class Init,
            class Pass,
            class Prologue,
            class Epilogue,
            class Sizes,
            class Offsets,
            class Output,
            class Inputs,
            class MakeComposite,
            class MakeIterator>
        void naive_apply_scan(builtins::scan<IsBackward, Init, Pass, Prologue, Epilogue>,
            Sizes const &sizes,
            Offsets const &offsets,
            Output &output,
            Inputs &&inputs,
            MakeComposite &&make_composite,
            MakeIterator &&make_iterator) {
            using in_tags_t = meta::transform<in_tag, meta::make_indices<tuple_util::size<Inputs>, std::tuple>>;
            auto composite = std::forward<MakeComposite>(make_composite)(out_tag(), in_tags_t(), output, inputs);
            auto strides = sid::get_strides(composite);
            auto ptr = sid::get_origin(composite)();
            sid::multi_shift(ptr, strides, offsets);

            static_assert(has_key<Sizes, Vertical>());
            auto v_size = at_key<Vertical>(sizes);

            constexpr size_t min_v_size = 1 + meta::length<Prologue>() + meta::length<Epilogue>();
            assert(v_size >= min_v_size);
            auto const &v_stride = sid::get_stride<Vertical>(strides);

            if constexpr (IsBackward::value)
                sid::shift(ptr, v_stride, v_size - integral_constant<int, 1>());

            using step_t = integral_constant<int, IsBackward::value ? -1 : 1>;

            size_t n = v_size - min_v_size;

            make_loops(remove_key<Vertical>(sizes))([&](auto ptr, auto const &strides) {
                auto inc = [&] { sid::shift(ptr, v_stride, step_t()); };
                auto first = [&]<auto Get, auto F>(meta::val<F, Get>) {
                    auto res = std::apply(F, tuple_util::transform(make_iterator(ptr, strides), in_tags_t()));
                    *at_key<out_tag>(ptr) = Get(res);
                    inc();
                    return res;
                };
                auto next = [&]<auto Get, auto F>(auto acc, meta::val<F, Get>) {
                    auto res = std::apply([acc = std::move(acc)](auto const &...its) { return F(acc, its...); },
                        tuple_util::transform(make_iterator(ptr, strides), in_tags_t()));
                    *at_key<out_tag>(ptr) = Get(res);
                    inc();
                    return res;
                };
                auto acc = tuple_util::fold(next, first(Init()), meta::rename<std::tuple, Prologue>());
                for (size_t i = 0; i != n; ++i)
                    acc = next(std::move(acc), Pass());
                tuple_util::fold(next, std::move(acc), meta::rename<std::tuple, Epilogue>());
            })(ptr, strides);
        }
    } // namespace naive_apply_impl_
    using naive_apply_impl_::naive_apply;
    using naive_apply_impl_::naive_apply_scan;
} // namespace gridtools::fn
