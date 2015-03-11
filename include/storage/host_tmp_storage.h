#pragma once

#include "base_storage.h"
#include <boost/mpl/int.hpp>

/**
   @file
   @brief This file contains the implementation of a storage used for the 'block' strategy

   Blocking is a technique allowing to tune the execution in order to efficently 
   exploit caches. The goal is to reduce the computational and memory consumption 
   due to the presence of temporary intermediate fields, which are computed only 
   as an intermediate result to be used in the nect computation. The idea is the 
   following: instead of looping over the whole domain, proceed blockwise, decomposing 
   the domain in tiles. For each tiles we perform all the stages of the stencil. 

   This allows us to store the intermediate temporary fields in a storage which only 
   has the dimension of the tile, and not of the whole domain. Furthermore the tiles 
   can be defined small enough to fit into caches. It is thus required the definition 
   of an extra storage, which contains a subset of the original storage fields.

   The memory layout and access pattern is thus redefined in this class, where a 
   'local' numeration is defined. The data dependency between tiles produces the 
   urge of an 'halo' region, i.e. an overlap between the tiles. The storage access 
   is performed via an index. The usual 1-to-1 relation to pass from the index 
   \f$ID\f$ to the coordinates \f$c1, c2, c3\f$, involving the strides 
   \f$s1 > s2 > 1\f$, is as follows:

   \f[ID= c1*s1+c2*s2+c3\f]

   while each index identifies three coordinates as follow

   \f[c3=ID\%s2\f]

   \f[c2=\frac{ID\%s1-c3}{s2}\f]

   \f[c1=\frac{ID-c2-c3}{s1}\f]

   where the \f$\%\f$ operator defines the integer remainder of the division.

   This can be extended to higher dimensions and can be rewritten as a
   recurrency formula (implemented via recursion).
*/

namespace gridtools {

    /**@brief Temporary storage class for the blocked algorithm
       This storage contains one tile (see explanation in the file description) plus the halo region.
       \targ Backend is normally the Host backend
       \targ ValueType is the scalar type (usually doubles)
       \targ Layout is the storage layout, defining which dimension corresponds to which stride
       \targ TlieI is the tile dimension in the x direction
       \targ TileJ is the tile dimension in the y direction
       \targ MinusI is the halo dimension in the x direction at the left side
       \targ MinusJ is the halo dimension in the y direction at the bottom side
       \targ PlusI is the halo dimension in the x direction at the right side
       \targ PlusJ is the halo dimension in the y direction at the top side
    */
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
    struct host_tmp_storage : public base_storage<Backend
                                                  , ValueType
                                                  , Layout
                                                  , true
                                                  >
    {

        typedef base_storage<Backend
                             , ValueType
                             , Layout
                             , true> base_type;


        typedef typename base_type::layout layout;
        typedef typename base_type::value_type value_type;
        typedef typename base_type::iterator_type  iterator_type;
        typedef typename base_type::const_iterator_type const_iterator_type;

        typedef static_int<MinusI> minusi;
        typedef static_int<MinusJ> minusj;
        typedef static_int<PlusI> plusi;
        typedef static_int<PlusJ> plusj;

        using base_type::m_strides;
        using base_type::is_set;

        static const std::string info_string;

        uint_t m_halo[3];
        uint_t m_initial_offsets[3];

        /**
           constructor of the temporary storage.

           \param initial_offset_i
           \param initial_offset_j
           \param dim3
           \param \optional n_i_threads (Default 1)
           \param \optional n_j_threasd (Default 1)
           \param \optional init (Default value_type())
           \param \optional s (Default "default_name")
         */
        explicit host_tmp_storage(uint_t initial_offset_i,
                                  uint_t initial_offset_j,
                                  uint_t dim3,
                                  uint_t n_i_threads=1,
                                  uint_t n_j_threads=1,
                                  value_type init = value_type(),
                                  char const* s = "default name" )
            : base_type((TileI+MinusI+PlusI)*n_i_threads,(TileJ+MinusJ+PlusJ)*n_j_threads, dim3, init, s)
        {
            m_halo[0]=MinusI;
            m_halo[1]=MinusJ;
            m_halo[2]=0;
            m_initial_offsets[0] = initial_offset_i;
            m_initial_offsets[1] = initial_offset_j;
            m_initial_offsets[2] = 0 /* initial_offset_k*/;
        }


        host_tmp_storage() {}

        virtual ~host_tmp_storage() {}


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

        /**@brief increment of 1 step along the specified direction. This method is used to increment in the vertical direction, where at present no blocking is performed.*/
        template <uint_t Coordinate>
        GT_FUNCTION
        void increment(uint_t b, uint_t* index){
            base_type::template increment<Coordinate>(b, index);
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
        template <uint_t Coordinate>
        GT_FUNCTION
        void increment(uint_t steps, uint_t b, uint_t* index){
            // no blocking along k
            if(Coordinate != 2)
                {
                    uint_t tile=Coordinate==0?TileI:TileJ;
                    uint_t var=steps - b * tile;

                    uint_t coor=var-
                        m_initial_offsets[layout::template at_<Coordinate>::value] 
                        + m_halo[layout::template at_<Coordinate>::value];

                    BOOST_STATIC_ASSERT(layout::template at_<Coordinate>::value>=0);
                    *index += coor*m_strides[layout::template at_<Coordinate>::value+1];
                }
            else
                {
                    base_type::template increment<Coordinate>( steps, b, index);
                }
        }

        /**@brief decrement in the horizontal direction (i or j). Analogous to the increment.
           TODO avoid code repetition*/
        template <uint_t Coordinate>
        GT_FUNCTION
        void decrement(uint_t& steps, uint_t& b, uint_t* index){

            uint_t tile=Coordinate==0?TileI:TileJ;
            uint_t var=steps - b * tile;
            BOOST_STATIC_ASSERT(layout::template at_<Coordinate>::value>=0);
            uint_t coor=var-m_initial_offsets[layout::template at_<Coordinate>::value] + m_halo[layout::template find<Coordinate>::value];
            *index -= coor*m_strides[layout::template at_<Coordinate>+1];
        }
    };


    template < enumtype::backend Backend, typename ValueType, typename Layout, uint_t TileI, uint_t TileJ, uint_t MinusI, uint_t MinusJ, uint_t PlusI, uint_t PlusJ
               >
    const std::string host_tmp_storage<Backend, ValueType, Layout, TileI, TileJ, MinusI, MinusJ, PlusI, PlusJ>
    ::info_string=boost::lexical_cast<std::string>(minusi::value)+
                                                 boost::lexical_cast<std::string>(minusj::value)+
                                                 boost::lexical_cast<std::string>(plusi::value)+
                                                 boost::lexical_cast<std::string>(plusj::value);

    //################# below there are template specializations #########################

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
