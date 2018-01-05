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

#include <boost/mpl/bool.hpp>
#include "../common/array.hpp"
#include "../common/defs.hpp"

namespace gridtools {

    // TODO
    // - can someone add an explanation for these two types? Should they be the same or not?
    // - change the name of this header into something that describes both or split them
    // - this type is only needed in icosahedral (if I understand correctly), however the offset_computation (which is
    // common) is using it, seems like a refactoring is needed to decouple this strange dependency

    using grid_position_type = array< uint_t, 4 >;
    template < typename T >
    struct is_grid_position_type : boost::mpl::false_ {};
    // in the unlikely case where you want to differentiate between array<int_t,4> and grid_position_type
    // you are doomed
    template <>
    struct is_grid_position_type< grid_position_type > : boost::mpl::true_ {};

    using position_offset_type = array< int_t, 4 >;
    template < typename T >
    struct is_position_offset_type : boost::mpl::false_ {};
    template <>
    struct is_position_offset_type< position_offset_type > : boost::mpl::true_ {};
}
