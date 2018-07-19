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

#include <type_traits>

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/type_traits.hpp"

#ifdef STRUCTURED_GRIDS
#include "expandable_parameters/iterate_domain_expandable_parameters.hpp"
#else
#include "icosahedral_grids/iterate_domain_expandable_parameters.hpp"
#endif
#include "functor_decorator.hpp"
#include "hasdo.hpp"

namespace gridtools {
    namespace _impl {
        template <ushort_t ID, typename Functor, typename Interval>
        struct call_repeated {
            template <class IterateDomain>
            GT_FUNCTION static void call_do_method(IterateDomain &it_domain) {

                typedef conditional_t<
                    std::is_same<Interval, typename Functor::f_with_default_interval::default_interval>::value,
                    conditional_t<has_do<typename Functor::f_type, Interval>::value,
                        typename Functor::f_type,
                        typename Functor::f_with_default_interval>,
                    typename Functor::f_type>
                    functor_t;

                functor_t::template Do<iterate_domain_expandable_parameters<IterateDomain, ID> &>(
                    *static_cast<iterate_domain_expandable_parameters<IterateDomain, ID> *>(&it_domain), Interval{});

                call_repeated<ID - 1, Functor, Interval>::call_do_method(it_domain);
            }
        };

        template <typename Functor, typename Interval>
        struct call_repeated<0, Functor, Interval> {
            template <class T>
            GT_FUNCTION static void call_do_method(T &&) {}
        };
    } // namespace _impl

    template <class FunctorDecorator, class Interval, class IterateDomain>
    GT_FUNCTION void call_repeated(IterateDomain &iterate_domain) {
        GRIDTOOLS_STATIC_ASSERT(is_functor_decorator<FunctorDecorator>::value, GT_INTERNAL_ERROR);
        _impl::call_repeated<FunctorDecorator::repeat_t::value, FunctorDecorator, Interval>::call_do_method(
            iterate_domain);
    }

} // namespace gridtools
