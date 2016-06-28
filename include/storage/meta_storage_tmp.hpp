/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once
#include "meta_storage_base.hpp"

/**
   @file
   implementation of a class handling the storage meta information in case of temporary storage
   when using the block strategy
*/
namespace gridtools {

#ifndef CXX11_ENABLED
    template < uint_t Tile, uint_t Plus, uint_t Minus >
    struct tile;

    template < typename MetaStorageBase, typename TileI, typename TileJ >
    struct meta_storage_tmp;
#endif

    /**
       @class
       @brief specialization for the temporary storages and block strategy
    */
    template < typename MetaStorageBase,
        typename FirstTile,
#ifdef CXX11_ENABLED
        typename... Tiles
#else
        uint_t Tile,
        uint_t Plus,
        uint_t Minus
#endif
        >
    struct meta_storage_tmp
#ifndef CXX11_ENABLED
        < MetaStorageBase, FirstTile, tile< Tile, Plus, Minus > >
#endif
        : public MetaStorageBase {

        GRIDTOOLS_STATIC_ASSERT(is_meta_storage< MetaStorageBase >::type::value, "wrong type");
        GRIDTOOLS_STATIC_ASSERT(MetaStorageBase::is_temporary == true, "wrong type");

        static const bool is_temporary = true;
        typedef MetaStorageBase super;

#ifdef CXX11_ENABLED
        typedef meta_storage_tmp< MetaStorageBase, FirstTile, Tiles... > this_type;
        typedef typename boost::mpl::vector< FirstTile, Tiles... > tiles_vector_t;
#else
        typedef tile< Tile, Plus, Minus > TileJ;
        typedef meta_storage_tmp< MetaStorageBase, FirstTile, TileJ > this_type;
        typedef typename boost::mpl::vector2< FirstTile, TileJ > tiles_vector_t;
#endif
        typedef typename super::type basic_type;
        typedef typename super::layout layout;

        // loss of generality: here we suppose tiling in i-j
        typedef typename boost::mpl::at_c< tiles_vector_t, 0 >::type::s_tile_t tile_i_t;
        typedef typename boost::mpl::at_c< tiles_vector_t, 0 >::type::s_plus_t plus_i_t;
        typedef typename boost::mpl::at_c< tiles_vector_t, 0 >::type::s_minus_t minus_i_t;
        typedef typename boost::mpl::at_c< tiles_vector_t, 1 >::type::s_tile_t tile_j_t;
        typedef typename boost::mpl::at_c< tiles_vector_t, 1 >::type::s_plus_t plus_j_t;
        typedef typename boost::mpl::at_c< tiles_vector_t, 1 >::type::s_minus_t minus_j_t;

        static const uint_t tile_i = tile_i_t::value;
        static const uint_t plus_i = plus_i_t::value;
        static const uint_t minus_i = minus_i_t::value;
        static const uint_t tile_j = tile_j_t::value;
        static const uint_t plus_j = plus_j_t::value;
        static const uint_t minus_j = minus_j_t::value;

      private:
        array< uint_t, 3 > m_initial_offsets;

      public:
        /**
           @brief constructor

           @param initial_offset_i the initial global i coordinate of the ij block
           @param initial_offset_j the initial global j coordinate of the ij block
           @param dim3 the dimension in k direction
           @param n_i_threads number of threads in the i direction
           @param n_j_threads number of threads in the j direction

           This constructor creates a storage tile with one peace assigned to each thread.
           The partition of the storage in tiles is a strategy to enhance data locality.
         */
        constexpr meta_storage_tmp(uint_t const &initial_offset_i,
            uint_t const &initial_offset_j,
            uint_t const &dim3,
            uint_t const &n_i_threads = 1,
            uint_t const &n_j_threads = 1)
            : super((tile_i + minus_i + plus_i) * n_i_threads, (tile_j + minus_j + plus_j) * n_j_threads, dim3)
#ifdef CXX11_ENABLED
              ,
              m_initial_offsets {
            initial_offset_i - minus_i, initial_offset_j - minus_j, 0
        }
#endif
        {
#ifndef CXX11_ENABLED
            m_initial_offsets[0] = initial_offset_i - minus_i;
            m_initial_offsets[1] = initial_offset_j - minus_j;
            m_initial_offsets[2] = 0;
#endif
        }

        // copy ctor
        __device__ constexpr meta_storage_tmp(meta_storage_tmp const &other)
            : super(other), m_initial_offsets(other.m_initial_offsets) {}

        constexpr meta_storage_tmp() : super() {}

      public:
        virtual ~meta_storage_tmp() {}

        /**
           @brief returns the index (in the array of data snapshots) corresponding to the specified offset

           It returns offset unless it is negative or it exceeds the size of the internal array of snapshots. In the latter case it returns offset modulo the size of the array.
           In the former case it returns the array size's complement of -offset.
        */
        GT_FUNCTION
        static constexpr ushort_t get_index(short_t const &offset) { return super::get_index(offset); }

        /** @brief increment in the horizontal direction (i or j).
            This method updates the storage index, so that an increment
            of 'steps' is obtained in the 'Coordinate' direction.

            The formula for incrementing the indices is the following:

            given the coordinate direction \f$C\in\{0,1,2\}\f$, the index i
            defining the increment in the direction C, and the global
            storage index ID, which identifies univocally the current
            storage entry and has to be updated with the increment, :

            \f$ID=ID+i-(b*tile)-offset+halo\f$

            where tile is the tile dimension in the C direction, b
            is the current block index being accessed, offset an halo
            are respectively the constant offset at the domain boundary
            for the coordinate C and the dimension of the overlap along
            C between tiles (identified by the data dependency
            requirements between tiles).
        */
        template < uint_t Coordinate, enumtype::execution Execution, typename StridesVector >
        GT_FUNCTION static void increment(int_t *RESTRICT index_, StridesVector const &RESTRICT strides_) {
            super::template increment< Coordinate, Execution >(index_, strides_);
        }

        template < uint_t Coordinate, typename StridesVector >
        GT_FUNCTION static void increment(
            const int_t &steps_, int_t *RESTRICT index_, StridesVector const &RESTRICT strides_) {
            super::template increment< Coordinate >(steps_, index_, strides_);
        }

        /**
           @brief initializing a given coordinate (i.e. multiplying times its stride)

           \param steps_ the input coordinate value
           \param block_ the current block index
           \param index_ the output index
           \param strides_ the strides array

           NOTE: this method is not static, while it is in the non-temporary case
        */
        template < uint_t Coordinate, typename StridesVector >
        GT_FUNCTION void initialize(
            const int_t steps_, const uint_t block_, int_t *RESTRICT index_, StridesVector const &strides_) const {

            // no blocking along k
            if (Coordinate != 2) {
                uint_t tile_ = Coordinate == 0 ? tile_i : tile_j;
                BOOST_STATIC_ASSERT(layout::template at_< Coordinate >::value >= 0);
                *index_ += (steps_ - block_ * tile_ - m_initial_offsets[Coordinate]) *
                           basic_type::template strides< Coordinate >(strides_);
            } else {
                super::template initialize< Coordinate >(steps_, block_, index_, strides_);
            }
        }

        /**
           index is the index in the array of field pointers, as defined in the base_storage

           The EU stands for ExecutionUnit (thich may be a thread or a group of
           threasd. There are potentially two ids, one over i and one over j, since
           our execution model is parallel on (i,j). Defaulted to 1.
        */
        GT_FUNCTION
        uint_t fields_offset(int_t EU_id_i, int_t EU_id_j) const {
            return (super::template strides< 0 >(super::strides())) * (tile_i + minus_i + plus_i) * EU_id_i +
                   (super::template strides< 1 >(super::strides())) * (tile_j + minus_j + plus_j) * EU_id_j;
        }
    };

    template < typename T >
    struct is_meta_storage;

#ifdef CXX11_ENABLED
    template < typename MetaStorageBase, typename... Tiles >
    struct is_meta_storage< meta_storage_tmp< MetaStorageBase, Tiles... > > : boost::mpl::true_ {};
#else
    template < typename MetaStorageBase, typename TileI, typename TileJ >
    struct is_meta_storage< meta_storage_tmp< MetaStorageBase, TileI, TileJ > > : boost::mpl::true_ {};
#endif

} // namespace gridtools
