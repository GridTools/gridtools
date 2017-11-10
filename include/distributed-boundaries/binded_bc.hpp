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

#pragma once

#include <tuple>
#include "../boundary-conditions/boundary.hpp"

namespace gridtools {
    template < typename BCApply, typename... DataStores >
    struct binded_bc {
        using boundary_class = BCApply;
        boundary_class m_bcapply;
        using stores_type = std::tuple< DataStores... >;
        stores_type m_stores;

        binded_bc(BCApply bca, DataStores... stores_list) : m_bcapply{bca}, m_stores{stores_list...} {}

        stores_type stores() { return m_stores; }
        stores_type const stores() const { return m_stores; }
        boundary_class boundary_to_apply() const { return m_bcapply; }
    };

    template < typename BCApply, typename... DataStores >
    binded_bc< BCApply, DataStores... > bind_bc(BCApply bc_apply, DataStores &... stores) {
        return {bc_apply, stores...};
    }

    template < typename T >
    struct is_binded_bc {
        static constexpr bool value = false;
    };

    template < typename... T >
    struct is_binded_bc< binded_bc< T... > > {
        static constexpr bool value = true;
    };

} // namespace gridtools
