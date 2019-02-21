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
#include "direction.hpp"

/**
@file
@brief This file contains the most common predicates used for the boundary condition assignment.
The predicates identify a regoin given a @ref gridtools::direction and its data members.
*/

namespace gridtools {

    /** \ingroup Boundary-Conditions
     * @{
     */

    /** @brief Default predicate that returns always true, so that the boundary conditions are applied everywhere
     */
    struct default_predicate {
        template <typename Direction>
        bool operator()(Direction) const {
            return true;
        }
    };

    /** @} */

} // namespace gridtools
