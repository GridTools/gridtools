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

#include "../../common/gt_assert.hpp"
#include "../common/definitions.hpp"

namespace gridtools {

    /** \ingroup storage
     * @{
     */

    /**
     *  @brief A class that represents the state machine that is used to determine
     *  if a storage is currently on the host or on the device and if the
     *  data on the host or the device is outdated and needs to be updated.
     */
    class state_machine {
        enum class state { synced, invalid_host, invalid_device };
        state m_state;

      public:
        state_machine() : m_state(state::synced) {}

        // actions
        void touch_host(access_mode_read_only_t) const {}
        void touch_host(access_mode_read_write_t = {}) {
            GT_ASSERT_OR_THROW(!host_needs_update(),
                "There is already an active read-write device view. Synchronization is needed before constructing the "
                "view.");
            m_state = state::invalid_device;
        }
        void touch_device(access_mode_read_only_t) const {}
        void touch_device(access_mode_read_write_t = {}) {
            GT_ASSERT_OR_THROW(!device_needs_update(),
                "There is already an active read-write host view. Synchronization is needed before constructing the "
                "view.");
            m_state = state::invalid_host;
        }

        // queries
        bool host_needs_update() const { return m_state == state::invalid_host; }
        bool device_needs_update() const { return m_state == state::invalid_device; }
    };

    /**
     * @}
     */
} // namespace gridtools
