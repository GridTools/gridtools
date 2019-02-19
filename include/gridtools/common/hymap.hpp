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

#ifndef GT_TARGET_ITERATING
//// DON'T USE #pragma once HERE!!!
#ifndef GT_COMMON_HYMAP_HPP_
#define GT_COMMON_HYMAP_HPP_

#include "../meta.hpp"
#include "defs.hpp"
#include "generic_metafunctions/utility.hpp"
#include "host_device.hpp"
#include "integral_constant.hpp"
#include "tuple_util.hpp"

namespace gridtools {

    template <class Keys, class Values>
    struct hymap : Values {
        GT_STATIC_ASSERT(tuple_util::size<Values>::value == meta::length<Keys>::value, "invalid hymap");

        using Values::Values;

        friend GT_META_CALL(tuple_util::traits::to_types, Values) tuple_to_types(hymap);

        struct from_types_f {
            template <class... Types>
#if GT_BROKEN_TEMPLATE_ALIASES
            struct apply {
                using type =
                    hymap<Keys, typename tuple_util::traits::from_types<Values>::type::template apply<Types...>::type>;
            };
#else
            using apply = hymap<Keys, typename tuple_util::traits::from_types<Values>::template apply<Types...>>;
#endif
        };

        friend from_types_f tuple_from_types(hymap);

        friend Keys hymap_get_keys(hymap);
    };

    namespace hymap_impl_ {

        template <class I>
        GT_META_DEFINE_ALIAS(get_key, meta::id, (integral_constant<int, I::value>));

        template <class T>
        GT_META_DEFINE_ALIAS(default_keys,
            meta::transform,
            (get_key, GT_META_CALL(meta::make_indices_for, GT_META_CALL(tuple_util::traits::to_types, T))));

        struct not_provided;

        not_provided hymap_get_keys(...);

        template <class T, class Res = decltype(hymap_get_keys(std::declval<T const &>()))>
        enable_if_t<!std::is_same<Res, not_provided>::value, Res> get_keys_fun(T const &);

        template <class T, class Res = decltype(hymap_get_keys(std::declval<T const &>()))>
        enable_if_t<std::is_same<Res, not_provided>::value, default_keys<T>> get_keys_fun(T const &);

        template <class T>
        using get_keys = decltype(::gridtools::hymap_impl_::get_keys_fun(std::declval<T const &>()));
    } // namespace hymap_impl_

#if GT_BROKEN_TEMPLATE_ALIASES
    template <class T>
    struct get_keys : meta::id<hymap_impl_::get_keys<T>> {};
#else
    using hymap_impl_::get_keys;
#endif

    template <class Map, class Key>
    GT_META_DEFINE_ALIAS(has_key, meta::st_contains, (hymap_impl_::get_keys<Map>, Key));

} // namespace gridtools

#define GT_FILENAME <gridtools/common/hymap.hpp>
#include GT_ITERATE_ON_TARGETS()
#undef GT_FILENAME

#endif // GT_COMMON_HYMAP_HPP_
#else  // GT_TARGET_ITERATING

namespace gridtools {
    GT_TARGET_NAMESPACE {

        template <class Key,
            class Map,
            class I = GT_META_CALL(meta::st_position, (GT_META_CALL(get_keys, decay_t<Map>), Key))>
        GT_TARGET GT_FORCE_INLINE constexpr auto at_key(Map && map) noexcept GT_AUTO_RETURN(
            tuple_util::GT_TARGET_NAMESPACE_NAME::get<I::value>(const_expr::forward<Map>(map)));

        template <class Keys, template <class...> class L, class... Vals>
        GT_TARGET GT_FORCE_INLINE constexpr hymap<Keys, L<Vals...>> make_hymap(Vals const &... vals) {
            return {vals...};
        }
    }
} // namespace gridtools

#endif // GT_TARGET_ITERATING