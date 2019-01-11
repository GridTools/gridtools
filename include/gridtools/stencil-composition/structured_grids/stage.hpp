/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

/**
 *   @file
 *
 *   Stage concept represents elementary functor from the backend implementor point of view.
 *
 *   Stage must have the nested `extent_t` type or an alias that has to model Extent concept.
 *   The meaning: the stage should be computed in the area that is extended from the user provided computation aria by
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
#include "../arg.hpp"
#include "../expressions/expr_base.hpp"
#include "../hasdo.hpp"
#include "../iterate_domain_fwd.hpp"
#include "./extent.hpp"

namespace gridtools {

    namespace impl_ {
        template <class ItDomain, class Args>
        struct evaluator {
            GRIDTOOLS_STATIC_ASSERT((meta::all_of<is_plh, Args>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT(is_iterate_domain<ItDomain>::value, GT_INTERNAL_ERROR);

            ItDomain const &m_it_domain;

            template <typename Accessor>
            GT_FUNCTION auto operator()(Accessor const &arg) const
                GT_AUTO_RETURN((m_it_domain.template deref<GT_META_CALL(meta::at_c, (Args, Accessor::index_t::value)),
                                Accessor::intent>(arg)));

            template <class Op, class... Ts>
            GT_FUNCTION auto operator()(expr<Op, Ts...> const &arg) const
                GT_AUTO_RETURN(expressions::evaluation::value(*this, arg));

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
        GRIDTOOLS_STATIC_ASSERT(has_do<Functor>::value, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(is_extent<Extent>::value, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((meta::all_of<is_plh, Args>::value), GT_INTERNAL_ERROR);

        using extent_t = Extent;

        template <class ItDomain>
        static GT_FUNCTION void exec(ItDomain const &it_domain) {
            GRIDTOOLS_STATIC_ASSERT(is_iterate_domain<ItDomain>::value, GT_INTERNAL_ERROR);
            impl_::evaluator<ItDomain, Args> eval{it_domain};
            Functor::Do(eval);
        }
    };

    template <class Stage, class... Stages>
    struct compound_stage {
        using extent_t = typename Stage::extent_t;

        GRIDTOOLS_STATIC_ASSERT(sizeof...(Stages) != 0, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(
            (conjunction<std::is_same<typename Stages::extent_t, extent_t>...>::value), GT_INTERNAL_ERROR);

        template <class ItDomain>
        static GT_FUNCTION void exec(ItDomain const &it_domain) {
            GRIDTOOLS_STATIC_ASSERT(is_iterate_domain<ItDomain>::value, GT_INTERNAL_ERROR);
            Stage::exec(it_domain);
            (void)(int[]){((void)Stages::exec(it_domain), 0)...};
        }
    };
} // namespace gridtools
