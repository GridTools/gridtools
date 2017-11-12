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

#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/sort.hpp>
#include "level_metafunctions_fwd.hpp"

namespace gridtools {
    /**
     * @brief Meta function to compare two levels: left <= right
     */
    template < typename TLevelLeft, typename TLevelRight, typename Enable = void >
    struct level_leq : boost::mpl::false_ {};

    template < typename TLevelLeft, typename TLevelRight >
    struct level_leq< TLevelLeft,
        TLevelRight,
        typename std::enable_if< (level_to_index< TLevelLeft >::value <= level_to_index< TLevelRight >::value) >::type >
        : boost::mpl::true_ {};

    /**
     * @brief Meta function to compare two levels: left < right
     */
    template < typename TLevelLeft, typename TLevelRight, typename Enable = void >
    struct level_lt : boost::mpl::false_ {};

    template < typename TLevelLeft, typename TLevelRight >
    struct level_lt< TLevelLeft,
        TLevelRight,
        typename std::enable_if< (level_to_index< TLevelLeft >::value < level_to_index< TLevelRight >::value) >::type >
        : boost::mpl::true_ {};

    /**
     * @brief Meta function to compare two levels: left >= right
     */
    template < typename TLevelLeft, typename TLevelRight, typename Enable = void >
    struct level_geq : boost::mpl::false_ {};

    template < typename TLevelLeft, typename TLevelRight >
    struct level_geq< TLevelLeft,
        TLevelRight,
        typename std::enable_if< (level_to_index< TLevelLeft >::value >= level_to_index< TLevelRight >::value) >::type >
        : boost::mpl::true_ {};

    /**
     * @brief Meta function to compare two levels: left > right
     */
    template < typename TLevelLeft, typename TLevelRight, typename Enable = void >
    struct level_gt : boost::mpl::false_ {};

    template < typename TLevelLeft, typename TLevelRight >
    struct level_gt< TLevelLeft,
        TLevelRight,
        typename std::enable_if< (level_to_index< TLevelLeft >::value > level_to_index< TLevelRight >::value) >::type >
        : boost::mpl::true_ {};

    /**
     * @brief return sorted mpl::vector of levels (ascending order)
     */
    template < typename... Levels >
    using sort_levels =
        typename boost::mpl::sort< boost::mpl::vector< Levels... >, level_gt< boost::mpl::_, boost::mpl::_ > >::type;

} // namespace gridtools
