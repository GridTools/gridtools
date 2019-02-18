/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

namespace gridtools {

    /** \ingroup storage
     * @{
     */

    /**
     *  @brief A class that represents the state machine that is used to determine
     *  if a storage is currently on the host or on the device and if the
     *  data on the host or the device is outdated and needs to be updated.
     */
    struct state_machine {
        bool m_hnu; // hnu = host needs update, set to true if a non-read-only device view is instantiated.
        bool m_dnu; // dnu = device needs update, set to true if a non-read-only host view is instantiated.
    };

    /**
     * @}
     */
} // namespace gridtools
