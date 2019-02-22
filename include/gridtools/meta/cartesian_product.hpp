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

#include "concat.hpp"
#include "curry.hpp"
#include "fold.hpp"
#include "list.hpp"
#include "macros.hpp"
#include "push_back.hpp"
#include "rename.hpp"
#include "transform.hpp"

namespace gridtools {
    namespace meta {
        template <class L>
        struct cartesian_product_step_impl_impl {
            template <class T>
            GT_META_DEFINE_ALIAS(
                apply, transform, (curry<push_back, T>::template apply, GT_META_CALL(rename, (list, L))));
        };

        template <class S, class L>
        GT_META_DEFINE_ALIAS(cartesian_product_step_impl,
            rename,
            (concat, GT_META_CALL(transform, (cartesian_product_step_impl_impl<L>::template apply, S))));

        template <class... Lists>
        GT_META_DEFINE_ALIAS(cartesian_product, lfold, (cartesian_product_step_impl, list<list<>>, list<Lists...>));
    } // namespace meta
} // namespace gridtools
