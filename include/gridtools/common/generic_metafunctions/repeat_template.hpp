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
/**
   @file

   Metafunction for creating a template class with an arbitrary length template parameter pack.
*/

#include <type_traits>

#include "../../meta/macros.hpp"
#include "../../meta/push_front.hpp"
#include "../../meta/repeat.hpp"
#include "../defs.hpp"

namespace gridtools {

    namespace _impl {
        template <template <ushort_t...> class Lambda, class Args>
        struct apply_lambda;

        template <template <ushort_t...> class Lambda, template <class...> class L, class... Ts>
        struct apply_lambda<Lambda, L<Ts...>> {
            using type = Lambda<Ts::value...>;
        };
    } // namespace _impl

    /**
       @brief Metafunction for creating a template class with an arbitrary length template parameter pack.

       Usage example:
       I have a class template halo< .... >, and I want to fill it by repeating N times the same number H
       \verbatim
       repeat_template_c<H, N, halo>
       \endverbatim
       Optionally a set of initial values to start filling the template class can be passed
    */
    template <ushort_t Constant, ushort_t Length, template <ushort_t... T> class Lambda, ushort_t... InitialValues>
    struct repeat_template_c {
        using repeated_args_t = GT_META_CALL(meta::repeat_c, (Length, std::integral_constant<ushort_t, Constant>));
        using all_args_t = GT_META_CALL(
            meta::push_front, (repeated_args_t, std::integral_constant<ushort_t, InitialValues>...));
        using type = typename _impl::apply_lambda<Lambda, all_args_t>::type;
    };

} // namespace gridtools
