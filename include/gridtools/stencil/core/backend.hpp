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

#include <functional>
#include <utility>

#include "../../common/hymap.hpp"
#include "../../common/tuple.hpp"
#include "../../common/tuple_util.hpp"
#include "../../sid/sid_shift_origin.hpp"
#include "convert_fe_to_be_spec.hpp"

namespace gridtools {
    namespace stencil {
        namespace core {
            namespace backend_impl_ {
                template <class Grid, class DataStores>
                auto shift_origin(Grid const &grid, DataStores data_stores) {
                    return tuple_util::transform(
                        [offsets = grid.origin()](
                            auto &&src) { return sid::shift_sid_origin(std::forward<decltype(src)>(src), offsets); },
                        std::move(data_stores));
                }

                template <class Backend, class Spec>
                struct backend_entry_point_f {
                    template <class Grid, class DataStores>
                    void operator()(Grid const &grid, DataStores data_stores) const {
                        gridtools_backend_entry_point(Backend(),
                            convert_fe_to_be_spec<Spec, typename Grid::interval_t, DataStores>(),
                            grid,
                            shift_origin(grid, std::move(data_stores)));
                    }
                };
            } // namespace backend_impl_
            using backend_impl_::backend_entry_point_f;
        } // namespace core
    }     // namespace stencil
} // namespace gridtools
