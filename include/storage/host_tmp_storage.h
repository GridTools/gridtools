#pragma once

#include "base_storage.h"
#include <boost/mpl/int.hpp>

/**
@file
@brief This file contains the implementation of a storage used for the 'block' strategy
Blocking is a technique allowing to tune the execution in order to efficently exploit cashes. The goal is to reduce the computational and memory consumption due to the presence of temporary intermediate fields, which are computed only as an intermediate result to be used in the nect computation. The idea is the following: instead of looping over the whole domain, proceed blockwise, decomposing the domain in tiles. For each tiles we perform all the stages of the stencil. This allows us to store the intermediate temporary fields in a storage which only has the dimension of the tile, and not of the whole domain. Furthermore the tiles can be defined small enough to fit into caches. It is thus required the definition of an extra storage, which contains a subset of the original storage fields.
The memory layout and access pattern is thus redefined in this class, where a 'local' numeration is defined. The data dependency between tiles produces the urge of an 'halo' region, i.e. an overlap between the tiles. The storage access is performed via an index. The usual 1-to-1 relation to pass from the index \f$ID\f$ to the coordinates \f$c1, c2, c3\f$, involving the strides \f$s1 > s2 > 1\f$, is as follows:

\f$ID= c1*s1+c2*s2+c3\f$

while each index identifies three coordinates as follow

\f$c3=ID%s2\f$

\f$c2=\frac{ID%s1-c3}{s2}\f$

\f$c1=\frac{ID-c2-c3}{s1}\f$

where the % operator defines the integer remain of the division.
This can be extended to higher dimensions and can be rewritten as a recurrency formula (implemented via recursion).
*/

namespace gridtools {


    //N=0 is the coordinate with stride 1
    template <uint_t N>
    struct coord_from_index;
    //specializations outside the class scope

    template<>
    struct coord_from_index<2>
    {
        static uint_t apply(uint_t index, uint_t* strides){
            printf("the coord from index: tile along %d is %d\n ", 0, strides[2]);
            return index%strides[2];
//host_tmp_storage<Backend, ValueType, Layout, TileI, TileJ, MinusI, MinusJ, PlusI, PlusJ>::tile<0>::value
        }
    };

    template<>
    struct coord_from_index<1>
        {
            static uint_t apply(uint_t index, uint_t* strides){
                printf("the coord from index: tile along %d is %d\n ", 1, strides[1]);
                return (index%strides[1]// tile<N>::value
                        -index% strides[2]);//(index%(K*J)-index%K%base_type::size()
            }
        };


    template<>
    struct coord_from_index<0>
        {
            static uint_t apply(uint_t index, uint_t* strides){
                printf("the coord from index: tile along %d is %d\n ", 2, strides[0]);
                return (index//%strides[0]
                        -index%strides[1]-index% strides[2]);//(index%(K*J)-index%K
            }
        };


    template < enumtype::backend Backend
               , typename ValueType
               , typename Layout
               , uint_t TileI
               , uint_t TileJ
               , uint_t MinusI
               , uint_t MinusJ
               , uint_t PlusI
               , uint_t PlusJ
               >
    struct host_tmp_storage : public storage<base_storage<Backend
                                                          , ValueType
                                                          , Layout
                                                          , true
                                                          > >
    {

        typedef storage<base_storage<Backend
                                     , ValueType
                                     , Layout
                                     , true> > base_type;


        typedef typename base_type::super::layout layout;
        typedef typename base_type::super::value_type value_type;
        typedef typename base_type::super::iterator_type  iterator_type;
        typedef typename base_type::super::const_iterator_type const_iterator_type;

        typedef static_int<MinusI> minusi;
        typedef static_int<MinusJ> minusj;
        typedef static_int<PlusI> plusi;
        typedef static_int<PlusJ> plusj;

        using base_type::super::m_strides;
        using base_type::super::is_set;

        static const std::string info_string;

        uint_t m_halo[3];
        uint_t m_initial_offsets[3];

        // std::string m_name;

        explicit host_tmp_storage(uint_t initial_offset_i,
                                  uint_t initial_offset_j,
                                  uint_t dim3,
                                  //int initial_offset_k=0,
                                  value_type init = value_type(),
                                  std::string const& s = std::string("default name") )
            : base_type(TileI+MinusI+PlusI,TileJ+MinusJ+PlusJ, dim3, init, s)
            {
                m_halo[0]=MinusI;
                m_halo[1]=MinusJ;
                m_halo[2]=0;
                m_initial_offsets[0] = initial_offset_i;
                m_initial_offsets[1] = initial_offset_j;
                m_initial_offsets[2] = 0 /* initial_offset_k*/;
            }


        host_tmp_storage() {}

        // ~host_tmp_storage() {
        //     if (is_set) {
        //         //std::cout << "deleting " << std::hex << data << std::endl;
        //         delete[] m_data;
        //     }
        // }


        virtual void info() const {
            std::cout << "Temporary storage "
                      << m_halo[0] << "x"
                      << m_halo[1] << "x"
                      << m_halo[2] << ", "
                      << "Initial offset "
                      << m_initial_offsets[0] << "x"
                      << m_initial_offsets[1] << "x"
                      << m_initial_offsets[2] << ", "
                      << this->m_name
                      << std::endl;
        }

        uint_t move_to(uint_t i,uint_t j,uint_t k) const {
            return tmp_index(i,j,k);
            //return &(base_type::m_data[_index(i,j,k)]);
        }

        GT_FUNCTION
        value_type& operator()(uint_t i, uint_t j, uint_t k) {
            return base_type::super::m_data[tmp_index(i,j,k)];
        }


        GT_FUNCTION
        value_type const & operator()(uint_t i, uint_t j, uint_t k) const {
            return base_type::super::m_data[tmp_index(i,j,k)];
        }


        uint_t tmp_index(uint_t i, uint_t j, uint_t k) const {
            uint_t index;
            // std::cout << "                                                  index "
            //           << "m_dims_i "
            //           << m_dims[0]
            //           << " "
            //           << "m_dims_j "
            //           << m_dims[1]
            //           << " "
            //           << "m_dims_k "
            //           << m_dims[2]
            //           << " - "
            //           << "m_halo_i "
            //           << m_halo[0]
            //           << " "
            //           << "m_halo_j "
            //           << m_halo[1]
            //           << " "
            //           << "m_halo_k "
            //           << m_halo[2]
            //           << " - "
            //           << "i "
            //           << i
            //           << " "
            //           << "j "
            //           << j
            //           << " "
            //           << "k "
            //           << k
            //           << std::endl;
            // info();
            assert(false);
            uint_t _i = ((layout::template find<0>(i,j,k)) - layout::template find<0>(m_initial_offsets) + layout::template find<0>(m_halo));
            std::cout << "uint_t _i = ((" << layout::template find<0>(i,j,k) << ")-" << layout::template find<0>(m_initial_offsets) << "+" << layout::template find<0>(m_halo) << ")" << std::endl;
            uint_t _j = ((layout::template find<1>(i,j,k)) - layout::template find<1>(m_initial_offsets) + layout::template find<1>(m_halo));
            std::cout << "uint_t _j = ((" << layout::template find<1>(i,j,k) << ")-" << layout::template find<1>(m_initial_offsets) << "+" << layout::template find<1>(m_halo) << ")" << std::endl;
            uint_t _k = ((layout::template find<2>(i,j,k)) - layout::template find<2>(m_initial_offsets) + layout::template find<2>(m_halo));

            index =
                /*layout::template find<2>(m_dims) * layout::template find<1>(m_dims)*/m_strides[1] * _i +
                /*layout::template find<2>(m_dims)*/m_strides[2] * _j + _k;

            // index =
            //     /*layout::template find<2>(m_dims) * layout::template find<1>(m_dims)*/m_strides[1] * (i%4) +
            //     /*layout::template find<2>(m_dims)*/m_strides[2] * (j%4) + k;



            // std::cout << " i  = " << _i
            //           << " j  = " << _j
            //           << " k  = " << _k
            //           << "   = " << index
            //           << std::endl;

            assert(false);
            assert(index >= 0);
            assert(index <m_strides[0]);

            return index;
        }

        // GT_FUNCTION
        // value_type& operator()(uint_t i, uint_t j, uint_t k) {
        //     /* std::cout<<"indices= "<<i<<" "<<j<<" "<<k<<std::endl; */
        //     //backend_traits_t::assertion(_index(i,j,k) >= 0);
        //     backend_traits_t::assertion(_index(i,j,k) < /* m_size*/ m_strides[0]);
        //     return m_data[_index(i,j,k)];
        // }


        // GT_FUNCTION
        // value_type const & operator()(uint_t i, uint_t j, uint_t k) const {
        //     //backend_traits_t::assertion(_index(i,j,k) >= 0);
        //     backend_traits_t::assertion(_index(i,j,k) < /*m_size*/m_strides[0]);
        //     return m_data[_index(i,j,k)];
        // }

        /**@brief prints a single value of the data field given the coordinates*/
        void print_value(uint_t i, uint_t j, uint_t k){ printf("value(%d, %d, %d)=%f, at index %d on the data\n", i, j, k, base_type::super::m_data[tmp_index(i, j, k)], tmp_index(i, j, k));}

        //N=0 is the dimension with stride 1
        template <uint_t N>
        struct tile
        {
            static const uint_t value=layout::template find<N, boost::mpl::vector_c<uint_t, (TileI+MinusI+PlusI)*(TileJ+MinusJ+PlusJ), (TileJ+MinusJ+PlusJ)*6,6> >();
        };

        // template <uint_t Coordinate>
        // GT_FUNCTION
        // void increment(uint_t& b, uint_t* index){
        //     //printf("NO blocking on K; index before incrementing: %d\n", *index);
        //     uint_t idx = *index + (base_type::template strides<Coordinate>(m_strides));
        //     //printf("Nidx: %d\n", idx);
        //     // uint_t coord=coord_from_index<layout::template pos_<Coordinate>::value >::apply(idx, m_strides);//this is e.g. j*strideJ
        //     // assert(coord%base_type::template strides<Coordinate>(m_strides)==0);
        //     // printf("coords before division: %d\n", coord);
        //     // coord=coord/base_type::template strides<Coordinate>(m_strides);// this is e.g. j
        //     // printf("coords after division: %d\n", coord);
        //     // uint dim=base_type::template dims_stridewise<Coordinate>(m_strides);
        //     // printf("k loop index: %d; idx: %d, coord: %d, dim: %d  \n", *index, idx, coord, base_type::template dims_stridewise<Coordinate>(m_strides));
        //     // if( coord <= dim)
        //     *index=idx;
        //     // else
        //     //     *index=*index-((coord-1)%dim)*base_type::template strides<Coordinate>(m_strides);

        //     // printf("index : %d -> %d, in direction %d, which in stride order means %d\n", base_type::template strides<Coordinate>(m_strides),  *index, Coordinate, layout::template pos_<Coordinate>::value);
        // }

        // template <uint_t Coordinate>
        // GT_FUNCTION
        // void decrement(uint_t& b, uint_t* index){
        //     *index = (*index - base_type::template strides<Coordinate>(m_strides))%m_strides[0];//%(Coordinate==0?TileI+MinusI+PlusI:TileJ+MinusJ+PlusJ);
        // }

        template <uint_t Coordinate>
        GT_FUNCTION
        void increment(uint_t b, uint_t* index){
            base_type::super::template increment<Coordinate>(b, index);
        }

        template <uint_t Coordinate>
        GT_FUNCTION
        void increment(uint_t dimension, uint_t b, uint_t* index){
            std::cout<<"dimension: "<<(int)dimension<<"block: "<<(int)b<<"tile: "<<(int) (Coordinate?TileI:TileJ) << std::endl;
            uint_t tile=Coordinate==0?TileI:TileJ;
            uint_t var=dimension - b * tile;

            std::cout << "uint_t coor = ((" << var << ")-" << layout::template find<Coordinate>(&m_initial_offsets[0]) << "+" << layout::template find<Coordinate>(&m_halo[0]) << ")";
            uint_t coor=var-layout::template find<Coordinate>(&m_initial_offsets[0]) + layout::template find<Coordinate>(&m_halo[0]);
            std::cout << " = "<<coor<<std::endl;
            *index += coor*layout::template find<Coordinate>(&m_strides[1]);
            std::cout << "index = "<<*index<<std::endl;
            std::cout << "strides = "<<m_strides[0]<<", "<<m_strides[1]<<", "<<m_strides[2] <<std::endl;
        }

        template <uint_t Coordinate>
        GT_FUNCTION
        void decrement(uint_t& dimension, uint_t& b, uint_t* index){

            uint_t tile=Coordinate==0?TileI:TileJ;
            uint_t var=dimension - b * tile;
            uint_t coor=var-layout::template find<Coordinate>(&m_initial_offsets[0]) + layout::template find<Coordinate>(&m_halo[0]);
            *index -= coor*layout::template find<Coordinate>(&m_strides[1]);
        }


    };


    template < enumtype::backend Backend, typename ValueType, typename Layout, uint_t TileI, uint_t TileJ, uint_t MinusI, uint_t MinusJ, uint_t PlusI, uint_t PlusJ
               >
    const std::string host_tmp_storage<Backend, ValueType, Layout, TileI, TileJ, MinusI, MinusJ, PlusI, PlusJ>
    ::info_string=boost::lexical_cast<std::string>(minusi::value)+
        boost::lexical_cast<std::string>(minusj::value)+
        boost::lexical_cast<std::string>(plusi::value)+
        boost::lexical_cast<std::string>(plusj::value);

    template <enumtype::backend Backend,
              typename ValueType
              , typename Layout
              , uint_t TileI
              , uint_t TileJ
              , uint_t MinusI
              , uint_t MinusJ
              , uint_t PlusI
              , uint_t PlusJ
              >
    std::ostream& operator<<(std::ostream& s,
                             host_tmp_storage<
                             Backend
                             , ValueType
                             , Layout
                             , TileI
                             , TileJ
                             , MinusI
                             , MinusJ
                             , PlusI
                             , PlusJ
                             > const & x) {
        return s << "host_tmp_storage<...,"
                 << TileI << ", "
                 << TileJ << ", "
                 << MinusI << ", "
                 << MinusJ << ", "
                 << PlusI << ", "
                 << PlusJ << "> ";
    }


    template <enumtype::backend Backend
              , typename ValueType
              , typename Layout
              , uint_t TileI
              , uint_t TileJ
              , uint_t MinusI
              , uint_t MinusJ
              , uint_t PlusI
              , uint_t PlusJ
              >
    struct is_storage<host_tmp_storage<
                          Backend
                          , ValueType
                          , Layout
                          , TileI
                          , TileJ
                          , MinusI
                          , MinusJ
                          , PlusI
                          , PlusJ
                          >* >
    : boost::false_type
    {};


    template <enumtype::backend Backend
              , typename ValueType
              , typename Layout
              , uint_t TileI
              , uint_t TileJ
              , uint_t MinusI
              , uint_t MinusJ
              , uint_t PlusI
              , uint_t PlusJ
              >
    struct is_temporary_storage<host_tmp_storage<
                                    Backend
                                    , ValueType
                                    , Layout
                                    , TileI
                                    , TileJ
                                    , MinusI
                                    , MinusJ
                                    , PlusI
                                    , PlusJ
                                    >*& >
    : boost::true_type
    {};

    template <enumtype::backend Backend
              , typename ValueType
              , typename Layout
              , uint_t TileI
              , uint_t TileJ
              , uint_t MinusI
              , uint_t MinusJ
              , uint_t PlusI
              , uint_t PlusJ
              >
    struct is_temporary_storage<host_tmp_storage<
                                    Backend,
                                    ValueType
                                    , Layout
                                    , TileI
                                    , TileJ
                                    , MinusI
                                    , MinusJ
                                    , PlusI
                                    , PlusJ
                                    >* >
    : boost::true_type
    {};

    template <enumtype::backend Backend
              , typename ValueType
              , typename Layout
              , uint_t TileI
              , uint_t TileJ
              , uint_t MinusI
              , uint_t MinusJ
              , uint_t PlusI
              , uint_t PlusJ
              >
    struct is_temporary_storage<host_tmp_storage<
                                    Backend
                                    , ValueType
                                    , Layout
                                    , TileI
                                    , TileJ
                                    , MinusI
                                    , MinusJ
                                    , PlusI
                                    , PlusJ
                                    > &>
    : boost::true_type
    {};

    template <enumtype::backend Backend
              , typename ValueType
              , typename Layout
              , uint_t TileI
              , uint_t TileJ
              , uint_t MinusI
              , uint_t MinusJ
              , uint_t PlusI
              , uint_t PlusJ
              >
    struct is_temporary_storage<host_tmp_storage<
                                    Backend
                                    , ValueType
                                    , Layout
                                    , TileI
                                    , TileJ
                                    , MinusI
                                    , MinusJ
                                    , PlusI
                                    , PlusJ
                                    > const& >
    : boost::true_type
    {};

} // namespace gridtools
