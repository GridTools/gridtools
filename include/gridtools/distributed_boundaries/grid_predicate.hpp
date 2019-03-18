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

#include "../boundary_conditions/direction.hpp"

namespace gridtools {
    /** @brief predicate returning whether I am or not at the global boundary, based on a processor grid
     */
    template <typename ProcGrid>
    struct proc_grid_predicate {
        ProcGrid const &m_grid;

        proc_grid_predicate(ProcGrid const &g) : m_grid{g} {}

        template <sign I, sign J, sign K>
        bool operator()(direction<I, J, K>) const {
            return (m_grid.template proc<I, J, K>() == -1);
        }
    };
} // namespace gridtools
