#pragma once
#include "meta_storage.hpp"

namespace gridtools{

    template<ushort_t Index, typename Layout, typename First, typename ... Tiles>
    struct meta_storage_base<Index, Layout, true, First, Tiles...> : public meta_storage_base<Index, Layout, false> {
        static const bool is_temporary=true;
        typedef  meta_storage_base<Index, Layout, false> super;

        typedef meta_storage_base<Index, Layout, true, First, Tiles ...> this_type;
        typedef typename super::basic_type basic_type;
        typedef typename super::layout layout;

        typedef typename boost::mpl::vector<First, Tiles ...> tiles_vector_t;

        //loss of generality: here we suppose tiling in i-j
        typedef typename boost::mpl::at_c<tiles_vector_t, 0>::type::s_tile_t  tile_i_t;
        typedef typename boost::mpl::at_c<tiles_vector_t, 0>::type::s_plus_t  plus_i_t;
        typedef typename boost::mpl::at_c<tiles_vector_t, 0>::type::s_minus_t  minus_i_t;
        typedef typename boost::mpl::at_c<tiles_vector_t, 1>::type::s_tile_t  tile_j_t;
        typedef typename boost::mpl::at_c<tiles_vector_t, 1>::type::s_plus_t  plus_j_t;
        typedef typename boost::mpl::at_c<tiles_vector_t, 1>::type::s_minus_t  minus_j_t;

        static const uint_t tile_i   = tile_i_t::value;
        static const uint_t plus_i   = plus_i_t::value;
        static const uint_t  minus_i = minus_i_t::value;
        static const uint_t tile_j   = tile_j_t::value;
        static const uint_t plus_j   = plus_j_t::value;
        static const uint_t  minus_j = minus_j_t::value;

    private:
        array<uint_t, 3> m_initial_offsets;

    public:

        constexpr meta_storage_base( uint_t const& initial_offset_i,
                                    uint_t const& initial_offset_j,
                                    uint_t const& dim3,
                                    uint_t const& n_i_threads=1,
                                    uint_t const& n_j_threads=1)
            : super((tile_i+minus_i+plus_i)*n_i_threads,(tile_j+minus_j+plus_j)*n_j_threads, dim3/*, init, s*/)
            , m_initial_offsets{initial_offset_i - minus_i, initial_offset_j - minus_j, 0}
            {}



        //copy ctor
        __device__
        constexpr meta_storage_base(meta_storage_base const& other)
            :  super(other), m_initial_offsets(other.m_initial_offsets)
        {
        }

        constexpr meta_storage_base() :super() {}

    public:
        virtual ~meta_storage_base() {}


        /**
           @brief returns the index (in the array of data snapshots) corresponding to the specified offset
           basically it returns offset unless it is negative or it exceeds the size of the internal array of snapshots. In the latter case it returns offset modulo the size of the array.
           In the former case it returns the array size's complement of -offset.
        */
        GT_FUNCTION
        static constexpr ushort_t get_index (short_t const& offset) {
            return super::get_index(offset);
        }



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
        template <uint_t Coordinate, enumtype::execution Execution, typename StridesVector>
        GT_FUNCTION
        static void increment( int_t* RESTRICT index_, StridesVector const& RESTRICT strides_){
            super::template increment<Coordinate, Execution>( index_, strides_);
        }

        template <uint_t Coordinate, typename StridesVector>
        GT_FUNCTION
        static void increment(const int_t& steps_, int_t* RESTRICT index_, StridesVector const&  RESTRICT strides_){
            super::template increment<Coordinate>( steps_, index_, strides_);
        }


        /*this one is not static*/
        template <uint_t Coordinate, typename StridesVector>
        GT_FUNCTION
        void initialize(const int_t steps_, const uint_t block_, int_t* RESTRICT index_, StridesVector const& strides_) const {

            // no blocking along k
            if(Coordinate != 2)
            {
                uint_t tile_=Coordinate==0?tile_i:tile_j;
                BOOST_STATIC_ASSERT(layout::template at_<Coordinate>::value>=0);
                *index_+=(steps_ - block_*tile_ - m_initial_offsets[Coordinate])*basic_type::template strides<Coordinate>(strides_);
            }
            else
            {
                super::template initialize<Coordinate>( steps_, block_, index_, strides_);
            }
        }

    };


}//namespace gridtools
