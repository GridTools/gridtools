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
#include "../../common/gt_assert.hpp"
#include "../../common/generic_metafunctions/gt_integer_sequence.hpp"
#include "../../common/array.hpp"
#include "../block_size.hpp"
#include "../extent.hpp"
#include "../../common/offset_tuple.hpp"

#ifdef CXX11_ENABLED
#include "meta_storage_cache.hpp"
#include "cache_storage_metafunctions.hpp"
#endif

namespace gridtools {
    template < typename T, typename U >
    struct get_storage_accessor;

    template < typename BlockSize, typename Extent, uint_t NColors, typename Storage >
    struct cache_storage;

#ifdef CXX11_ENABLED
    /**
     * @struct cache_storage
     * simple storage class for storing caches. Current version is multidimensional, but allows the user only to cache
     * an entire dimension.
     * Which dimensions to cache is decided by the extents. if the extent is 0,0 the dimension is not cached (and CANNOT
     * BE ACCESSED with an offset other than 0).
     * In a cached data field we suppose that all the snapshots get cached (adding an extra dimension to the
     * meta_storage_cache)
     * in future version we need to support K and IJK storages. Data is allocated on the stack.
     * The size of the storage is determined by the block size and the extension to this block sizes required for
     *  halo regions (determined by a extent type)
     * @tparam Value value type being stored
     * @tparam BlockSize physical domain block size
     * @tparam Extend extent
     * @tparam NColors number of colors of the location type of the storage
     * @tparam Storage type of the storage
     */
    template < uint_t... Tiles, short_t... ExtentBounds, typename Storage, uint_t NColors >
    struct cache_storage< block_size< Tiles... >, extent< ExtentBounds... >, NColors, Storage > {

      public:
        typedef typename unzip< variadic_to_vector< static_short< ExtentBounds >... > >::first minus_t;
        typedef typename unzip< variadic_to_vector< static_short< ExtentBounds >... > >::second plus_t;
        typedef variadic_to_vector< static_int< Tiles >... > tiles_t;

        // Storage must be a gridtools::pointer to storage
        GRIDTOOLS_STATIC_ASSERT(is_pointer< Storage >::value, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(is_storage< typename Storage::value_type >::value, GT_INTERNAL_ERROR);
        typedef typename Storage::value_type::basic_type storage_t;
        typedef typename storage_t::value_type value_type;

        // generate a layout map with the number of dimensions of the tiles + 1(snapshots) + 1 (field dimension)
        typedef typename _impl::generate_layout_map< typename make_gt_integer_sequence< uint_t,
            sizeof...(Tiles) + 2 /*FD*/
// TODO ICO_STORAGE in irregular grids we have one more dim for color
#ifndef STRUCTURED_GRIDS
                +
                1
#endif
            >::type >::type layout_t;

        GT_FUNCTION
        explicit constexpr cache_storage() {}

        typedef
            typename _impl::compute_meta_storage< layout_t, plus_t, minus_t, tiles_t, NColors, storage_t >::type meta_t;

        GT_FUNCTION
        static constexpr uint_t size() { return meta_t{}.size(); }

        template < uint_t Color, typename Accessor >
        GT_FUNCTION value_type &RESTRICT at(array< int, 2 > const &thread_pos, Accessor const &accessor_) {
            constexpr const meta_t s_storage_info;

            using accessor_t = typename boost::remove_const< typename boost::remove_reference< Accessor >::type >::type;
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< accessor_t >::value), GT_INTERNAL_ERROR_MSG("Error type is not accessor tuple"));

            typedef typename boost::mpl::at_c< typename minus_t::type, 0 >::type iminus;
            typedef typename boost::mpl::at_c< typename minus_t::type, 1 >::type jminus;

#ifdef CUDA8
            typedef static_int< s_storage_info.template strides< 0 >() > check_constexpr_1;
            typedef static_int< s_storage_info.template strides< 1 >() > check_constexpr_2;
#else
            assert((_impl::compute_size< NColors, minus_t, plus_t, tiles_t, storage_t >::value == size()));
#endif

            // manually aligning the storage
            const uint_t extra_ = (thread_pos[0] - iminus::value) * s_storage_info.template strides< 0 >() +
// TODO ICO_STORAGE
#ifdef STRUCTURED_GRIDS
                                  (thread_pos[1] - jminus::value) * s_storage_info.template strides< 1 >() +
#else
                                  Color * s_storage_info.template strides< 1 >() +
                                  (thread_pos[1] - jminus::value) * s_storage_info.template strides< 2 >() +
#endif
                                  s_storage_info.index(accessor_);
            assert((extra_) < size());
            assert((extra_) >= 0);

            return m_values[extra_];
        }

      private:
#if defined(CUDA8)
        value_type m_values[size()];
#else

        value_type m_values[_impl::compute_size< NColors, minus_t, plus_t, tiles_t, storage_t >::value];
#endif
    };

#else // CXX11_ENABLED

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
    template < typename BlockSize, typename Extend, uint_t NColors, typename Storage >
    struct cache_storage {

        GRIDTOOLS_STATIC_ASSERT((is_block_size< BlockSize >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_extent< Extend >::value), GT_INTERNAL_ERROR);

        typedef typename BlockSize::i_size_t tile_i;
        typedef typename BlockSize::j_size_t tile_j;
        typedef typename Extend::iminus iminus;
        typedef typename Extend::jminus jminus;
        typedef typename Extend::iplus iplus;
        typedef typename Extend::jplus jplus;

        typedef static_uint< 1 > i_stride_t;
        typedef static_uint< tile_i::value - iminus::value + iplus::value > c_stride_t;
        typedef static_uint< c_stride_t::value * NColors > j_stride_t;
        typedef static_uint< j_stride_t::value *(tile_j::value - jminus::value + jplus::value) > storage_size_t;

        typedef typename Storage::value_type::basic_type storage_t;
        typedef typename storage_t::value_type value_type;

        explicit cache_storage() {}

        template < uint_t Color, typename Offset >
        GT_FUNCTION value_type &RESTRICT at(array< int, 2 > const &thread_pos, Offset const &offset) {
            GRIDTOOLS_STATIC_ASSERT((is_offset_tuple< typename Offset::offset_tuple_t >::value),
                GT_INTERNAL_ERROR_MSG("Error type is not offset tuple"));
            assert(index< Color >(thread_pos, offset.offsets()) < storage_size_t::value);
            assert(index< Color >(thread_pos, offset.offsets()) >= 0);

            return m_values[index< Color >(thread_pos, offset.offsets())];
        }

      private:
        template < uint_t Color, typename Offset >
        GT_FUNCTION int_t index(array< int, 2 > const &thread_pos, Offset const &offset) {
            return (thread_pos[0] + offset.template get< Offset::n_args - 1 >() - iminus::value) * i_stride_t::value +
// TODO ICO_STORAGE
#ifdef STRUCTURED_GRIDS
                   (thread_pos[1] + offset.template get< Offset::n_args - 2 >() - jminus::value) * j_stride_t::value;
#else
                   (Color + offset.template get< Offset::n_args - 2 >()) * c_stride_t::value +
                   (thread_pos[1] + offset.template get< Offset::n_args - 3 >() - jminus::value) * j_stride_t::value;
#endif
        }

        value_type m_values[storage_size_t::value];
    };
#endif // CXX11_ENABLED

} // namespace gridtools
