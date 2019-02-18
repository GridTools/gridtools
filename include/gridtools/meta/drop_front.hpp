/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <cstddef>

#include "defs.hpp"
#include "id.hpp"
#include "list.hpp"
#include "macros.hpp"
#include "repeat.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  Drop N elements from the front of the list
         *
         *  Complexity is amortized O(1).
         */
        GT_META_LAZY_NAMESPACE {
            template <class SomeList, class List>
            class drop_front_impl;
            template <class... Us, template <class...> class L, class... Ts>
            class drop_front_impl<list<Us...>, L<Ts...>> {
                template <class... Vs>
                static L<Vs...> select(Us *..., id<Vs> *...);

              public:
                using type = decltype(select(((id<Ts> *)0)...));
            };

            template <class N, class List>
            GT_META_DEFINE_ALIAS(drop_front, drop_front_impl, (typename repeat_c<N::value, void>::type, List));

            template <std::size_t N, class List>
            GT_META_DEFINE_ALIAS(drop_front_c, drop_front_impl, (typename repeat_c<N, void>::type, List));
        }
#if !GT_BROKEN_TEMPLATE_ALIASES
        template <std::size_t N, class List>
        using drop_front_c = typename lazy::drop_front_impl<typename repeat_c<N, void>::type, List>::type;
        template <class N, class List>
        using drop_front = typename lazy::drop_front_impl<typename repeat_c<N::value, void>::type, List>::type;
#endif
    } // namespace meta
} // namespace gridtools
