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

#include "../../common/defs.hpp"
#include "../../common/integral_constant.hpp"
#include "../../common/stride_util.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "synthetic.hpp"

namespace gridtools {
    namespace sid {
        namespace contiguous_impl_ {
            struct make_zero_f {
                template <class T>
                integral_constant<int_t, 0> operator()(T &&) const {
                    return {};
                }
            };

            template <class Sizes>
            auto zeros(Sizes const &sizes) GT_AUTO_RETURN(tuple_util::transform(make_zero_f{}, sizes));

            template <class Sizes>
            using strides_type = decltype(stride_util::make_strides_from_sizes(std::declval<Sizes const &>()));

        } // namespace contiguous_impl_

        template <class T, class PtrDiff = ptrdiff_t, class StridesKind = void, class Allocator, class Sizes>
        auto make_contiguous(Allocator &allocator, Sizes const &sizes) GT_AUTO_RETURN(
            (synthetic()
                    .set<property::origin>(allocate(allocator, meta::lazy::id<T>{}, stride_util::total_size(sizes)))
                    .template set<property::strides>(stride_util::make_strides_from_sizes(sizes))
                    .template set<property::ptr_diff, PtrDiff>()
                    .template set<property::strides_kind,
                        GT_META_CALL(meta::if_,
                            (std::is_void<StridesKind>, contiguous_impl_::strides_type<Sizes>, StridesKind))>()
                    .template set<property::upper_bounds>(sizes)
                    .template set<property::lower_bounds>(contiguous_impl_::zeros(sizes))));
    } // namespace sid
} // namespace gridtools
