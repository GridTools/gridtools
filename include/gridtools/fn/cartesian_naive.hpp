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

#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "../sid/allocator.hpp"
#include "../sid/composite.hpp"
#include "../sid/concept.hpp"
#include "../sid/contiguous.hpp"
#include "../sid/sid_shift_origin.hpp"
#include "ast.hpp"
#include "backend/naive.hpp"
#include "cartesian.hpp"
#include "fencil.hpp"
#include "naive_apply.hpp"
#include "strided_iter.hpp"

namespace gridtools::fn {
    namespace cartesian_naive_impl_ {
        template <class...>
        struct get_offset;
        template <class Dim>
        struct get_offset<Dim, meta::val<>> : integral_constant<int, 0> {};
        template <class Dim, auto D, auto V, auto... Vs>
        struct get_offset<Dim, meta::val<D, V, Vs...>> : get_offset<Dim, meta::val<Vs...>> {};
        template <class Dim, Dim D, auto V, auto... Vs>
        struct get_offset<Dim, meta::val<D, V, Vs...>>
            : integral_constant<int, V + get_offset<Dim, meta::val<Vs...>>::value> {};

        template <class Dim, class OffsetsList>
        using get_offsets =
            meta::transform<meta::curry<meta::force<get_offset>::apply, Dim>::template apply, OffsetsList>;

        template <template <class...> class L, class... Ts>
        constexpr auto get_min(L<Ts...>) {
            if constexpr (sizeof...(Ts))
                return std::min({Ts::value...});
            else
                return 0;
        }

        template <template <class...> class L, class... Ts>
        constexpr auto get_max(L<Ts...>) {
            if constexpr (sizeof...(Ts))
                return std::max({Ts::value...});
            else
                return 0;
        }

        template <class OffsetsList>
        struct extend_size_f {
            template <class Dim, class Val>
            auto operator()(Val val) const {
                using offsets_t = get_offsets<Dim, OffsetsList>;
                using min_t = integral_constant<int, get_min(offsets_t())>;
                using max_t = integral_constant<int, get_max(offsets_t())>;
                return val + max_t() - min_t();
            }
        };

        template <class OffsetsList, class Sizes>
        auto get_sizes(Sizes const &sizes) {
            return hymap::transform(extend_size_f<OffsetsList>(), sizes);
        }

        template <class OffsetsList>
        struct shift_offsets_f {
            template <class Dim, class Val>
            auto operator()(Val val) const {
                using offsets_t = get_offsets<Dim, OffsetsList>;
                using min_t = integral_constant<int, get_min(offsets_t())>;
                return val + min_t();
            }
        };

        template <class OffsetsList, class Offsets>
        auto get_domain_offsets(Offsets const &offsets) {
            return hymap::transform(shift_offsets_f<OffsetsList>(), offsets);
        }

        template <class OffsetsList>
        constexpr auto domain_extender = [](auto const &domain) {
            return cartesian(get_sizes<OffsetsList>(domain.sizes), get_domain_offsets<OffsetsList>(domain.offsets));
        };

        template <class Spec, class Allocator, class Domain>
        auto make_tmp(Allocator &allocator, Domain const &domain) {
            using value_t = meta::first<Spec>;
            using offsets_list_t = meta::second<Spec>;
            using kind_t = offsets_list_t;
            return sid::shift_sid_origin(
                sid::make_contiguous<value_t, int, kind_t>(allocator, get_sizes<offsets_list_t>(domain.sizes)),
                get_domain_offsets<offsets_list_t>(domain.offsets));
        }

        template <class Specs, class Allocator, class Domain>
        auto make_tmps(Allocator &allocator, Domain const &domain) {
            return tuple_util::transform(
                [&]<class Spec>(Spec) { return make_tmp<Spec>(allocator, domain); }, meta::rename<std::tuple, Specs>());
        }

        template <class Spec,
            class OffsetsList = meta::second<Spec>,
            class Stencil = meta::third<Spec>,
            class Out = meta::at_c<Spec, 3>,
            class Ins = meta::at_c<Spec, 4>>
        using make_stage_from_spec = stage<Stencil, meta::val<domain_extender<OffsetsList>>, Out, Ins>;

    } // namespace cartesian_naive_impl_
    template <auto Stencil, class Sizes, class Offsets, class Output, class... Inputs>
    void fn_apply(naive,
        cartesian<Sizes, Offsets> const &domain,
        meta::val<Stencil>,
        Output &output,
        std::tuple<Inputs...> inputs) {

        if constexpr (ast::has_tmps<Stencil, Inputs...>) {
            using input_types_t = meta::list<sid::element_type<Inputs>...>;
            using tree_t = ast::popup_tmps<ast::parse<Stencil, Inputs...>>;
            using specs_t = ast::flatten_tmps_tree<tree_t, input_types_t>;
            using stages_t = meta::transform<cartesian_naive_impl_::make_stage_from_spec, specs_t>;

            auto alloc = sid::make_allocator(&std::make_unique<char[]>);
            auto tmps = cartesian_naive_impl_::make_tmps<meta::pop_back<specs_t>>(alloc, domain);
            exec_stages(naive(), stages_t(), domain, tuple_util::push_back(tuple_util::concat(inputs, tmps), output));
        } else {
            naive_apply<Stencil>(domain.sizes,
                domain.offsets,
                output,
                std::move(inputs),
                []<class OutTag, class InTags, class Out, class Ins>(OutTag, InTags, Out & out, Ins && ins) {
                    using keys_t = meta::rename<sid::composite::keys, meta::push_back<InTags, OutTag>>;
                    return tuple_util::convert_to<keys_t::template values>(
                        tuple_util::push_back(std::forward<Ins>(ins), out));
                },
                [](auto &&ptr, auto const &strides) {
                    return [&](auto tag) { return strided_iter(tag, ptr, strides); };
                });
        }
    }
} // namespace gridtools::fn
