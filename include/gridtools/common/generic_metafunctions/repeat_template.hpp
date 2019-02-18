/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
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
