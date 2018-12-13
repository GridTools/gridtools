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

#include <type_traits>

#include "../../meta/utility.hpp"
#include "../defs.hpp"
#include "../host_device.hpp"

namespace gridtools {

    /** \ingroup common
        @{
        \ingroup allmeta
        @{
        \defgroup gtintegersequence GridTools Integer Sequence
        @{
    */

    /**
       @brief helper struct to use an integer sequence in order to fill a generic container

       can be used with an arbitrary container with elements of the same type (not a tuple),
       it is consexpr constructable.
     */
    template <typename UInt, UInt... Indices>
    using gt_integer_sequence = meta::integer_sequence<UInt, Indices...>;

    /** @brief constructs an integer sequence

        @tparam N size of the integer sequence
     */
    template <typename UInt, UInt N>
    using make_gt_integer_sequence = meta::make_integer_sequence<UInt, N>;

    template <size_t... Ints>
    using gt_index_sequence = meta::index_sequence<Ints...>;

    template <size_t N>
    using make_gt_index_sequence = meta::make_index_sequence<N>;

    template <class... Ts>
    using gt_index_sequence_for = meta::make_index_sequence<sizeof...(Ts)>;

    // with CXX14 the gt_integer_sequence from the standard can directly replace this one:
    // template <typename UInt, UInt ... Indices>
    // using gt_integer_sequence=std::integer_sequence<UInt, Indices ...>;

    // template<typename UInt, UInt N>
    // using make_gt_integer_sequence=std::make_integer_sequence<UInt, N>;

    /** @} */
    /** @} */
    /** @} */

} // namespace gridtools
