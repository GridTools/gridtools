/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <type_traits>

#include "concat.hpp"
#include "dedup.hpp"
#include "filter.hpp"
#include "first.hpp"
#include "list.hpp"
#include "mp_find.hpp"
#include "not.hpp"
#include "rename.hpp"
#include "transform.hpp"

namespace gridtools {
    namespace meta {
        template <template <class...> class F, class... Maps>
        struct mp_merge_helper_f {
            template <class Key>
            using apply = rename<F, filter<not_<std::is_void>::apply, list<mp_find<Maps, Key>...>>>;
        };

        template <template <class...> class F, class... Maps>
        using mp_merge =
            transform<mp_merge_helper_f<F, Maps...>::template apply, dedup<concat<transform<first, Maps>...>>>;
    } // namespace meta
} // namespace gridtools
