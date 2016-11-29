/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include "common/defs.hpp"
#include "common/storage_traits_metafunctions.hpp"
#include "storage_host/storage.hpp"
#include "storage_host/storage_info.hpp"
#include "storage_host/data_view_helpers.hpp"
#include "storage_host/data_field_view_helpers.hpp"

namespace gridtools {
    template < enumtype::platform T >
    struct storage_traits_from_id;

    /** @brief traits for the Host backend*/
    template <>
    struct storage_traits_from_id< enumtype::Host > {

        template < typename ValueType >
        struct select_storage {
            typedef host_storage< ValueType > type;
        };

        template < unsigned Id, unsigned Dims, typename Halo >
        struct select_storage_info {
            typedef typename get_layout< Dims, true >::type layout;
            typedef host_storage_info< Id, layout, Halo > type;
        };

        template < unsigned Id, typename Selector, typename Halo >
        struct select_special_storage_info {
            typedef typename get_layout< Selector::size, true >::type layout;
            typedef host_storage_info< Id, typename get_special_layout< layout, Selector >::type > type;
        };
    };
}
