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

#include "../common/gt_assert.hpp"
#include "grid_position_type.hpp"
#include <boost/mpl/eval_if.hpp>

namespace gridtools {

    namespace _impl {
        // AccessorOrPositionOffset is an accessor
        // TODO rename compute_offset_t to something that describes this computation
        // or at least add a comment to explain
        template < typename AccessorOrPositionOffset, unsigned N, typename Enable = void >
        struct compute_offset_t : boost::mpl::int_< ((AccessorOrPositionOffset::n_dimensions - 1) - N) > {};

        // AccessorOrPositionOffset is a position_offset
        template < typename AccessorOrPositionOffset, unsigned N >
        struct compute_offset_t< AccessorOrPositionOffset,
            N,
            typename std::enable_if< is_position_offset_type< AccessorOrPositionOffset >::value >::type >
            : boost::mpl::int_< N > {};

        template < typename Max, typename StridesCached, typename Accessor, typename StorageInfo, unsigned N >
        GT_FUNCTION constexpr
            typename boost::enable_if_c< (N == (StorageInfo::layout_t::masked_length - 1)), int_t >::type
            apply_accessor(StridesCached const &RESTRICT strides, Accessor const &RESTRICT acc) {
            typedef boost::mpl::int_< (StorageInfo::layout_t::template at< N >()) > val_t;
            GRIDTOOLS_STATIC_ASSERT((val_t::value == Max::value) || (N < StorageInfo::layout_t::masked_length),
                GT_INTERNAL_ERROR_MSG("invalid stride array access"));
            typedef boost::mpl::bool_< (val_t::value == Max::value) > is_max_t;
            typedef boost::mpl::bool_< (val_t::value == -1) > is_masked_t;
            typedef typename compute_offset_t< Accessor, N >::type offset_t;
            return (is_max_t::value ? 1 : (is_masked_t::value ? 0 : strides[(uint_t)val_t::value])) *
                   acc.template get< offset_t::value >();
        }

        template < typename Max, typename StridesCached, typename Accessor, typename StorageInfo, unsigned N >
        GT_FUNCTION constexpr
            typename boost::enable_if_c< (N < (StorageInfo::layout_t::masked_length - 1)), int_t >::type
            apply_accessor(StridesCached const &RESTRICT strides, Accessor const &RESTRICT acc) {
            typedef boost::mpl::int_< (StorageInfo::layout_t::template at< N >()) > val_t;
            GRIDTOOLS_STATIC_ASSERT((val_t::value == Max::value) || (N < StorageInfo::layout_t::masked_length),
                GT_INTERNAL_ERROR_MSG("invalid stride array access"));
            typedef boost::mpl::bool_< (StorageInfo::layout_t::template at< N >() == Max::value) > is_max_t;
            typedef boost::mpl::bool_< (StorageInfo::layout_t::template at< N >() == -1) > is_masked_t;
            typedef typename compute_offset_t< Accessor, N >::type offset_t;
            return (is_max_t::value ? 1 : (is_masked_t::value ? 0 : strides[(uint_t)val_t::value])) *
                       acc.template get< offset_t::value >() +
                   apply_accessor< Max, StridesCached, Accessor, StorageInfo, N + 1 >(strides, acc);
        }

        template < unsigned From = 0, unsigned To = 0, typename StorageInfo, typename Accessor >
        GT_FUNCTION typename boost::enable_if_c< (From == To), void >::type check_bounds_cache_offset_(Accessor acc) {
            return;
        }

        /** helper function checking if cache storage is being accesses out of bounds */
        template < unsigned From = 0, unsigned To = 0, typename StorageInfo, typename Accessor >
        GT_FUNCTION typename boost::enable_if_c< (From < To), void >::type check_bounds_cache_offset_(Accessor acc) {
            // check if accessin cache metastorage out of bounds
            assert((acc.template get< Accessor::n_dimensions - 1 - From >() <= StorageInfo::template dim< From >()));
        }

        template < typename StorageInfo, typename Accessor >
        GT_FUNCTION void check_bounds_cache_offset(Accessor acc) {
            check_bounds_cache_offset_< 0, StorageInfo::layout_t::masked_length, StorageInfo >(acc);
        }

        /** helper function (base case) computing sum(offset*stride ...) for cached fields */
        template < unsigned From = 0, unsigned To = 0, typename StorageInfo, typename Accessor >
        GT_FUNCTION constexpr typename boost::enable_if_c< (From == To), int_t >::type get_cache_offset_(Accessor acc) {
            return 0;
        }

        /** helper function (step case) computing sum(offset*stride ...) for cached fields */
        template < unsigned From = 0, unsigned To = 0, typename StorageInfo, typename Accessor >
        GT_FUNCTION constexpr typename boost::enable_if_c< (From < To), int_t >::type get_cache_offset_(Accessor acc) {
            return StorageInfo::template stride< From >() * acc.template get< Accessor::n_dimensions - 1 - From >() +
                   get_cache_offset_< From + 1, To, StorageInfo, Accessor >(acc);
        }
        /** helper function (step case) computing sum(offset*stride ...) for cached fields */
        template < typename StorageInfo, typename Accessor >
        GT_FUNCTION constexpr int_t get_cache_offset(Accessor acc) {
            return get_cache_offset_< 0, StorageInfo::layout_t::masked_length, StorageInfo >(acc);
        }
    }

    // pointer offset computation for uncached fields
    template < typename StorageInfo, typename Accessor, typename StridesCached >
    GT_FUNCTION constexpr int_t compute_offset(
        StridesCached const &RESTRICT strides_cached, Accessor const &RESTRICT acc) {
        // get the max coordinate of given StorageInfo
        typedef typename boost::mpl::deref< typename boost::mpl::max_element<
            typename StorageInfo::layout_t::static_layout_vector >::type >::type max_t;
        return _impl::apply_accessor< max_t, StridesCached, Accessor, StorageInfo, 0 >(strides_cached, acc);
    }
}
