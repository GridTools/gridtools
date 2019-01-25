/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
/*
 * @file
 * @brief file containing helper infrastructure, functors and metafunctions
 *  for the cache functionality of the iterate domain.
 */

#pragma once

#include <boost/fusion/include/at_key.hpp>

#include "../../common/array.hpp"
#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/host_device.hpp"
#include "../execution_types.hpp"

namespace gridtools {
    namespace _impl {

        template <enumtype::execution Policy, class Caches>
        struct slide_caches_f {
            Caches const &m_caches;

            template <class Arg>
            GT_FUNCTION void operator()(Arg) const {
                boost::fusion::at_key<Arg>(m_caches).template slide<Policy>();
            }
        };
        template <class Args, enumtype::execution Policy, class Caches>
        GT_FUNCTION void slide_caches(Caches const &caches) {
            host_device::for_each<Args>(slide_caches_f<Policy, Caches>{caches});
        }

        template <enumtype::execution Policy, sync_type SyncType, class ItDomain, class Caches>
        struct sync_caches_f {
            ItDomain const &m_it_domain;
            Caches const &m_caches;
            bool m_sync_all;
            array<int_t, 2> m_validity;

            template <class Arg>
            GT_FUNCTION void operator()(Arg) const {
                boost::fusion::at_key<Arg>(m_caches).template sync<Policy, SyncType>(
                    m_it_domain.template k_cache_deref<Arg>(), m_sync_all, m_validity);
            }
        };
        template <class Args, enumtype::execution Policy, sync_type SyncType, class ItDomain, class Caches>
        GT_FUNCTION void sync_caches(
            ItDomain const &it_domain, Caches const &caches, bool sync_all, array<int_t, 2> validity) {
            host_device::for_each<Args>(
                sync_caches_f<Policy, SyncType, ItDomain, Caches>{it_domain, caches, sync_all, validity});
        }

    } // namespace _impl
} // namespace gridtools
