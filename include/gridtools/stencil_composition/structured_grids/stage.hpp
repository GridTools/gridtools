/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 *   @file
 *
 *   Stage concept represents elementary functor from the backend implementor point of view.
 *
 *   Stage must have the nested `extent_t` type or an alias that has to model Extent concept.
 *   The meaning: the stage should be computed in the area that is extended from the user provided computation area by
 *   that much.
 *
 *   Stage also have static `exec` method that accepts an object by reference that models IteratorDomain.
 *   `exec` should execute an elementary functor in the grid point that IteratorDomain points to.
 *
 *   Note that the Stage is (and should stay) backend independent. The core of gridtools passes stages [split by k-loop
 *   intervals and independent groups] to the backend in the form of compile time only parameters.
 *
 *   TODO(anstaf): add `is_stage<T>` trait
 */

#pragma once

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../../meta/at.hpp"
#include "../../meta/logical.hpp"
#include "../../meta/macros.hpp"
#include "../../meta/type_traits.hpp"
#include "../accessor_intent.hpp"
#include "../arg.hpp"
#include "../expressions/expr_base.hpp"
#include "../has_apply.hpp"
#include "../iterate_domain_fwd.hpp"
#include "extent.hpp"

namespace gridtools {

    namespace impl_ {
        template <class ItDomain, class Args>
        struct evaluator {
            GT_STATIC_ASSERT((meta::all_of<is_plh, Args>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT(is_iterate_domain<ItDomain>::value, GT_INTERNAL_ERROR);

            ItDomain const &m_it_domain;

            template <class Accessor>
            GT_FUNCTION decltype(auto) operator()(Accessor const &arg) const {
                return apply_intent<Accessor::intent_v>(
                    m_it_domain.template deref<meta::at_c<Args, Accessor::index_t::value>>(arg));
            }

            template <class Op, class... Ts>
            GT_FUNCTION auto operator()(expr<Op, Ts...> const &arg) const {
                return expressions::evaluation::value(*this, arg);
            }

            GT_FUNCTION int_t i() const { return m_it_domain.i(); }
            GT_FUNCTION int_t j() const { return m_it_domain.j(); }
            GT_FUNCTION int_t k() const { return m_it_domain.k(); }
        };
    } // namespace impl_

    /**
     *   A stage that is associated with an elementary functor.
     */
    template <class Functor, class Extent, class Args>
    struct regular_stage {
        GT_STATIC_ASSERT(has_apply<Functor>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_extent<Extent>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((meta::all_of<is_plh, Args>::value), GT_INTERNAL_ERROR);

        using extent_t = Extent;

        template <class ItDomain>
        static GT_FUNCTION void exec(ItDomain const &it_domain) {
            GT_STATIC_ASSERT(is_iterate_domain<ItDomain>::value, GT_INTERNAL_ERROR);
            impl_::evaluator<ItDomain, Args> eval{it_domain};
            Functor::apply(eval);
        }
    };

    template <class Stage, class... Stages>
    struct compound_stage {
        using extent_t = typename Stage::extent_t;

        GT_STATIC_ASSERT(sizeof...(Stages) != 0, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((conjunction<std::is_same<typename Stages::extent_t, extent_t>...>::value), GT_INTERNAL_ERROR);

        template <class ItDomain>
        static GT_FUNCTION void exec(ItDomain const &it_domain) {
            GT_STATIC_ASSERT(is_iterate_domain<ItDomain>::value, GT_INTERNAL_ERROR);
            Stage::exec(it_domain);
            (void)(int[]){((void)Stages::exec(it_domain), 0)...};
        }
    };
} // namespace gridtools
