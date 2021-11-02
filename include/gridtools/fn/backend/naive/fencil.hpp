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

#include "../../../common/tuple_util.hpp"
#include "../../../meta.hpp"
#include "../../../sid/allocator.hpp"
#include "../../../sid/contiguous.hpp"
#include "../../../sid/sid_shift_origin.hpp"
#include "../../ast.hpp"
#include "../../domain_extender.hpp"
#include "tag.hpp"

namespace gridtools::fn {
    namespace naive_fencil_impl_ {
        template <template <class...> class L, class... Ts, class Refs>
        auto select(L<Ts...>, Refs const &refs) {
            return std::tie(std::get<Ts::value>(refs)...);
        }

        template <class Spec, class Allocator, class Domain>
        auto make_tmp(Allocator &allocator, Domain const &domain) {
            using value_t = meta::first<Spec>;
            using offsets_list_t = meta::second<Spec>;
            auto &&d = domain_extender<offsets_list_t>(domain);
            using kind_t = offsets_list_t;
            return sid::shift_sid_origin(sid::make_contiguous<value_t, int, kind_t>(allocator, d.sizes),
                tuple_util::transform(std::negate<>(), d.offsets));
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
        using make_stage_from_spec = meta::list<Stencil, meta::val<domain_extender<OffsetsList>>, Out, Ins>;

        template <auto Stencil, class Domain, class Output, class... Inputs>
        void process_stencil(Domain const &domain, Output &output, std::tuple<Inputs...> inputs) {
            if constexpr (ast::has_tmps<Stencil, Inputs...>) {
                using input_types_t = meta::list<sid::element_type<Inputs>...>;
                using tree_t = ast::popup_tmps<ast::parse<Stencil, Inputs...>>;
                using specs_t = ast::flatten_tmps_tree<tree_t, input_types_t>;
                using stages_t = meta::transform<make_stage_from_spec, specs_t>;
                auto alloc = sid::make_allocator(&std::make_unique<char[]>);
                auto tmps = make_tmps<meta::pop_back<specs_t>>(alloc, domain);
                fn_fencil(naive(), stages_t(), domain, tuple_util::push_back(tuple_util::concat(inputs, tmps), output));
            } else {
                fn_apply(naive(), domain, meta::constant<Stencil>, output, inputs);
            }
        }
    } // namespace naive_fencil_impl_

    template <template <class...> class L, class... Stages, class Domain, class Refs>
    void fn_fencil(naive, L<Stages...>, Domain const &domain, Refs const &refs) {
        using namespace naive_fencil_impl_;
        (...,
            process_stencil<meta::first<Stages>::value>(meta::second<Stages>::value(domain),
                std::get<meta::third<Stages>::value>(refs),
                select(meta::at_c<Stages, 3>(), refs)));
    }
} // namespace gridtools::fn
