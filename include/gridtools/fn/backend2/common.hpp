/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../../common/hymap.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta/rename.hpp"
#include "../../sid/loop.hpp"

namespace gridtools::fn::backend::common {

    template <class Sizes>
    GT_CONSTEXPR GT_FUNCTION auto make_loops(Sizes const &sizes) {
        return tuple_util::fold(
            [&](auto outer, auto dim) {
                return [outer = std::move(outer), inner = sid::make_loop<decltype(dim)>(at_key<decltype(dim)>(sizes))](
                           auto &&... args) { return outer(inner(std::forward<decltype(args)>(args)...)); };
            },
            identity(),
            meta::rename<tuple, get_keys<Sizes>>());
    }

} // namespace gridtools::fn::backend::common
