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

#include "../common/defs.hpp"

namespace gridtools {

    /**
     * @brief metadata with the information for architecture, grid and strategy backends
     * @tparam BackendId architecture backend id
     * @tparam StrategyId strategy id
     */
    template <class BackendId, class StrategyId>
    struct backend_ids {
        using strategy_id_t = StrategyId;
        using backend_id_t = BackendId;
    };

    template <typename T>
    struct is_backend_ids : boost::mpl::false_ {};

    template <class BackendId, class StrategyId>
    struct is_backend_ids<backend_ids<BackendId, StrategyId>> : boost::mpl::true_ {};
} // namespace gridtools
