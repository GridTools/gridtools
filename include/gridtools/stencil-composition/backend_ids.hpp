/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../common/defs.hpp"

namespace gridtools {

    /**
     * @brief metadata with the information for architecture, grid and strategy backends
     * @tparam BackendId architecture backend id
     * @tparam GridId grid backend id
     * @tparam StrategyId strategy id
     */
    template <class BackendId, class GridId, class StrategyId>
    struct backend_ids {
        using strategy_id_t = StrategyId;
        using backend_id_t = BackendId;
        using grid_id_t = GridId;
    };

    template <typename T>
    struct is_backend_ids : boost::mpl::false_ {};

    template <class BackendId, class GridId, class StrategyId>
    struct is_backend_ids<backend_ids<BackendId, GridId, StrategyId>> : boost::mpl::true_ {};
} // namespace gridtools
