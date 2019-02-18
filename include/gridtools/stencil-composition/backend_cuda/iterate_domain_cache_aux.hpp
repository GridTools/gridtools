/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
/*
 * @file
 * @brief file containing helper infrastructure, functors and metafunctions
 *  for the cache functionality of the iterate domain.
 */

#pragma once

#include <boost/fusion/include/at_key.hpp>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/host_device.hpp"
#include "../execution_types.hpp"

namespace gridtools {
    namespace _impl {

        template <typename Policy, class Caches>
        struct slide_caches_f {
            Caches &m_caches;

            template <class Arg>
            GT_FUNCTION void operator()() const {
                boost::fusion::at_key<Arg>(m_caches).template slide<Policy>();
            }
        };
        template <class Args, typename Policy, class Caches>
        GT_FUNCTION void slide_caches(Caches &caches) {
            host_device::for_each_type<Args>(slide_caches_f<Policy, Caches>{caches});
        }

        template <typename Policy, sync_type SyncType, class ItDomain, class Caches>
        struct sync_caches_f {
            ItDomain const &m_it_domain;
            Caches &m_caches;
            bool m_sync_all;

            template <class Arg>
            GT_FUNCTION void operator()() const {
                boost::fusion::at_key<Arg>(m_caches).template sync<Policy, SyncType>(m_it_domain, m_sync_all);
            }
        };
        template <class Args, typename Policy, sync_type SyncType, class ItDomain, class Caches>
        GT_FUNCTION void sync_caches(ItDomain const &it_domain, Caches &caches, bool sync_all) {
            host_device::for_each_type<Args>(
                sync_caches_f<Policy, SyncType, ItDomain, Caches>{it_domain, caches, sync_all});
        }

    } // namespace _impl
} // namespace gridtools
