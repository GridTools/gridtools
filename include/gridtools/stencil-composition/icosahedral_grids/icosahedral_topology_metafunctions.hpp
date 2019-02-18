/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "../../common/defs.hpp"
#include "../../common/gt_math.hpp"
#include "../../common/host_device.hpp"
#include "../../common/selector.hpp"

namespace gridtools {
    namespace impl {
        /**
         * @brief Computes a unique identifier (to be used for metastorages) given a list of index values
         */
        template <uint_t Pos>
        GT_FUNCTION constexpr long long compute_uuid_selector(int cnt) {
            return 0;
        }

        /**
         * @brief Computes a unique identifier (to be used for metastorages) given a list of index values
         */
        template <uint_t Pos, typename... Int>
        GT_FUNCTION constexpr long long compute_uuid_selector(int cnt, int val0, Int... val) {
            return (cnt == 4)
                       ? 0
                       : ((val0 == 1)
                                 ? gt_pow<Pos>::apply((long long)2) + compute_uuid_selector<Pos + 1>(cnt + 1, val...)
                                 : compute_uuid_selector<Pos + 1>(cnt + 1, val...));
        }

        /**
         * Computes a unique identifier (to be used for metastorages) given the location type and a dim selector
         * that determines the dimensions of the layout map which are activated.
         * Only the first 4 dimension of the selector are considered, since the iteration space of the backend
         * does not make use of indices beyond the space dimensions
         */
        template <int_t LocationTypeIndex, typename Selector>
        struct compute_uuid {};

        template <int_t LocationTypeIndex, bool... B>
        struct compute_uuid<LocationTypeIndex, selector<B...>> {
            static constexpr ushort_t value =
                metastorage_library_indices_limit + LocationTypeIndex + compute_uuid_selector<2>(0, B...);
        };
    } // namespace impl
} // namespace gridtools
