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
#include "../../common/gt_assert.hpp"
#include "../../common/array.hpp"
#include "../block_size.hpp"
#include "../extent.hpp"
#include "../../common/offset_tuple.hpp"

namespace gridtools {

    /**
     * @struct cache_storage
     * simple storage class for storing caches. Current version assumes only 2D (i,j), but it will be extented
     * in future version to support K and IJK storages. Data is allocated on the stack.
     * The size of the storage is determined by the block size and the extension to this block sizes required for
     *  halo regions (determined by a extent type)
     * @tparam Value value type being stored
     * @tparam BlockSize physical domain block size
     * @tparam Extend extent
     */
    template < typename Value, typename BlockSize, typename Extend, uint_t NColors >
    struct cache_storage {

        GRIDTOOLS_STATIC_ASSERT((is_block_size< BlockSize >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_extent< Extend >::value), "Internal Error: wrong type");

        typedef typename BlockSize::i_size_t tile_i;
        typedef typename BlockSize::j_size_t tile_j;
        typedef typename Extend::iminus iminus;
        typedef typename Extend::jminus jminus;
        typedef typename Extend::iplus iplus;
        typedef typename Extend::jplus jplus;

        typedef static_uint< 1 > i_stride_t;
        typedef static_uint< tile_i::value - iminus::value + iplus::value > c_stride_t;
        typedef static_uint< c_stride_t::value * NColors > j_stride_t;
        typedef static_uint< j_stride_t::value *
                             (tile_j::value - jminus::value + jplus::value)> storage_size_t;
        explicit cache_storage() {}

        template < uint_t Color, typename Offset >
        GT_FUNCTION Value &RESTRICT at(array< int, 2 > const &thread_pos, Offset const &offset) {
            GRIDTOOLS_STATIC_ASSERT((is_offset_tuple< Offset >::value), "Error type is not offset tuple");
            assert(index<Color>(thread_pos, offset) < storage_size_t::value);
            assert(index<Color>(thread_pos, offset) >= 0);

            return m_values[index<Color>(thread_pos, offset)];
        }

      private:
        template < uint_t Color, typename Offset >
        GT_FUNCTION int_t index(array< int, 2 > const &thread_pos, Offset const &offset) {
            return (thread_pos[0] + offset.template get< Offset::n_args - 1 >() - iminus::value) * i_stride_t::value +
                    (Color + offset.template get< Offset::n_args - 2 >())*c_stride_t::value +
//HACK
#ifdef STRUCTURED_GRIDS
                   (thread_pos[1] + offset.template get< Offset::n_args - 2 >() - jminus::value) * j_stride_t::value;
#else
                   (thread_pos[1] + offset.template get< Offset::n_args - 3 >() - jminus::value) * j_stride_t::value;
#endif
        }

        Value m_values[storage_size_t::value];
    };

} // namespace gridtools
