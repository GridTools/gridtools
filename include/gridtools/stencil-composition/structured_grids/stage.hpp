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

#pragma once

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/generic_metafunctions/meta.hpp"
#include "../../common/host_device.hpp"
#include "../arg.hpp"
#include "../expandable_parameters/iterate_domain_expandable_parameters.hpp"
#include "../iterate_domain_fwd.hpp"
#include "./extent.hpp"
#include "./iterate_domain_remapper.hpp"

namespace gridtools {

    namespace _impl {
        template <class Functor, class Eval>
        struct call_do_f {
            Eval *m_eval;
            template <class Index>
            GT_FUNCTION void operator()() const {
                using eval_t = iterate_domain_expandable_parameters<Eval, Index::value + 1>;
                Functor::template Do<eval_t &>(*reinterpret_cast<eval_t *>(m_eval));
            }
        };

    } // namespace _impl

    template <class Functor, class Extent, class Args, size_t RepeatFactor>
    struct regular_stage {
        GRIDTOOLS_STATIC_ASSERT(is_extent<Extent>::value, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((meta::all_of<is_arg, Args>::value), GT_INTERNAL_ERROR);

        using extent_t = Extent;

        template <class ItDomain>
        static GT_FUNCTION void exec(ItDomain &it_domain) {
            GRIDTOOLS_STATIC_ASSERT(is_iterate_domain<ItDomain>::value, GT_INTERNAL_ERROR);
            using eval_t = typename get_iterate_domain_remapper<ItDomain, Args>::type;
            eval_t eval{it_domain};
            for_each_type<GT_META_CALL(meta::make_indices_c, RepeatFactor)>(_impl::call_do_f<Functor, eval_t>{&eval});
        }
    };

    template <class Functor, class Extent, class Args, class BinOp>
    struct reduction_stage {
        GRIDTOOLS_STATIC_ASSERT(is_extent<Extent>::value, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((meta::all_of<is_arg, Args>::value), GT_INTERNAL_ERROR);

        using extent_t = Extent;

        template <class ItDomain>
        static GT_FUNCTION void exec(ItDomain &it_domain) {
            GRIDTOOLS_STATIC_ASSERT(is_iterate_domain<ItDomain>::value, GT_INTERNAL_ERROR);
            using eval_t = typename get_iterate_domain_remapper<ItDomain, Args>::type;
            eval_t eval{it_domain};
            it_domain.set_reduction_value(BinOp{}(it_domain.reduction_value(), Functor::template Do<eval_t &>(eval)));
        }
    };

} // namespace gridtools
