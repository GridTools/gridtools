#pragma once

#include "base_storage.h"

namespace gridtools {
    template < typename ValueType
               , typename Layout
               , int TileI
               , int TileJ
               , int MinusI
               , int MinusJ
               , int PlusI
               , int PlusJ
#ifndef NDEBUG
               , typename TypeTag = int
#endif
               >
    struct host_tmp_storage : public base_storage<host_tmp_storage<
                                                      ValueType
                                                      , Layout
                                                      , TileI
                                                      , TileJ
                                                      , MinusI
                                                      , MinusJ
                                                      , PlusI
                                                      , PlusJ
#ifndef NDEBUG
                                                      , TypeTag
#endif
                                                      >
                                                  , ValueType
                                                  , Layout
                                                  , true
                                                  > 
{

        typedef base_storage<host_tmp_storage<
                                 ValueType
                                 , Layout
                                 , TileI
                                 , TileJ
                                 , MinusI
                                 , MinusJ
                                 , PlusI
                                 , PlusJ
#ifndef NDEBUG
                                 , TypeTag
#endif
                                 >
                             ,ValueType
                             , Layout
                             , true> base_type;


        typedef Layout layout;
        typedef ValueType value_type;
        typedef value_type* iterator_type;
        typedef value_type const* const_iterator_type;

        using base_type::m_dims;
        using base_type::strides;
        using base_type::m_size;
        using base_type::is_set;

        int m_tile[3];
        int offs[3];

        std::string m_name;

        value_type* data;

        explicit host_tmp_storage(int m_dim3,
                                  value_type init = value_type(),
                                  std::string const& s = std::string("default name") )
            : base_type(TileI+MinusI+PlusI,TileJ+MinusJ+PlusJ, m_dim3, init)
            , m_name(s)
        {
            m_tile[0] = TileI;
            m_tile[1] = TileJ;
            m_tile[2] = m_dim3;
            offs[0]=MinusI;
            offs[1]=MinusJ;
            offs[2]=0;
            data = new value_type[m_size];
        }

        host_tmp_storage() {}

        ~host_tmp_storage() {
            if (is_set) {
                //std::cout << "deleting " << std::hex << data << std::endl;
                delete[] data;
            }
        }

        std::string const& name() const {
            return m_name;
        }

        static void text() {
            std::cout << BOOST_CURRENT_FUNCTION << std::endl;
        }

        value_type* min_addr() const {
            return data ;
        }

        value_type* max_addr() const {
            return data+m_size;
        }

        value_type& move_to(int i,int j,int k, int bi, int bj) const {
            return data[_index(i,j,k,bi,bj)];
        }

        // value_type& operator()(int i, int j, int k) {
        //     assert(_index(i,j,k) >= 0);
        //     assert(_index(i,j,k) < m_size);
        //     return data[_index(i,j,k)];
        // }

        // value_type const & operator()(int i, int j, int k) const {
        //     assert(_index(i,j,k) >= 0);
        //     assert(_index(i,j,k) < m_size);
        //     return data[_index(i,j,k)];
        // }

        // int offset(int i, int j, int k) const {
        //     return layout::template find<2>(m_dims) * layout::template find<1>(m_dims)
        //     * layout::template find<0>(i-offs_i,j-offs_j,k-offs_k) +
        //     layout::template find<2>(m_dims) * layout::template find<1>(i-offs_i,j-offs_j,k-offs_k) +
        //     layout::template find<2>(i-offs_i,j-offs_j,k-offs_k);
        // }

        virtual void info() const {
            std::cout << "Temporary storage "
                      << m_dims[0] << "x"
                      << m_dims[1] << "x"
                      << m_dims[2] << ", "
                      << offs[0] << "x"
                      << offs[1] << "x"
                      << offs[2] << ", "
                      << m_name
                      << std::endl;
        }
        
        int _index(int i, int j, int k) const {
            int index;
            std::cout << "                                                  index " 
                      << "offs_i "
                      << offs[0]
                      << " " 
                      << "offs_j "
                      << offs[1]
                      << " " 
                      << "offs_k "
                      << offs[2] 
                      << " - "
                      << "i "
                      << i 
                      << " " 
                      << "j "
                      << j 
                      << " " 
                      << "k "
                      << k 
                      << std::endl;
            info();


            index =
                layout::template find<2>(m_dims) * layout::template find<1>(m_dims)

                * ( ( (layout::template find<0>(i,j,k)-2) % layout::template find<0>(m_tile) ) 
                    + layout::template find<0>(offs) ) +

                layout::template find<2>(m_dims) * 

                ( ( (layout::template find<1>(i,j,k)-2) % layout::template find<1>(m_tile) )
                  + layout::template find<1>(offs)) +

                (layout::template find<2>(i,j,k) % layout::template find<2>(m_tile))
                 + layout::template find<2>(offs);


            std::cout
                // << "stride " << layout::template find<2>(m_dims) * layout::template find<1>(m_dims)
                // << " * " (modulus(layout::template find<0>(i,j,k)-2,layout::template find<0>(m_tile)) 
                //           + layout::template find<0>(offs)) << " + "
                // << "stride2 " <<  layout::template find<2>(m_dims) << " * " 
                // << (modulus(layout::template find<1>(i,j,k)-2,layout::template find<1>(m_tile))
                //     + layout::template find<1>(offs)) << " + "
                // << modulus(layout::template find<2>(i,j,k),layout::template find<2>(m_tile))
                // + layout::template find<2>(offs)
                << "   = " << index
                << std::endl;


            assert(index <m_size);
            assert(index >= 0);
            return index;
        }

    };
        
    template <typename ValueType
              , typename Layout
              , int TileI
              , int TileJ
              , int MinusI
              , int MinusJ
              , int PlusI
               , int PlusJ
#ifndef NDEBUG
              , typename TypeTag
#endif
              >
    struct is_temporary_storage<host_tmp_storage<
                                    ValueType
                                    , Layout
                                    , TileI
                                    , TileJ
                                    , MinusI
                                    , MinusJ
                                    , PlusI
                                    , PlusJ
#ifndef NDEBUG
                                    , TypeTag
#endif
                                    >*& >
      : boost::true_type
    {};

    template <typename ValueType
              , typename Layout
              , int TileI
              , int TileJ
              , int MinusI
              , int MinusJ
              , int PlusI
              , int PlusJ
#ifndef NDEBUG
              , typename TypeTag
#endif
              >
    struct is_temporary_storage<host_tmp_storage<
                                    ValueType
                                    , Layout
                                    , TileI
                                    , TileJ
                                    , MinusI
                                    , MinusJ
                                    , PlusI
                                    , PlusJ
#ifndef NDEBUG
                                    , TypeTag
#endif
                                    >* >
      : boost::true_type
    {};

    template <typename ValueType
              , typename Layout
              , int TileI
              , int TileJ
              , int MinusI
              , int MinusJ
              , int PlusI
              , int PlusJ
#ifndef NDEBUG
              , typename TypeTag
#endif
              >
    struct is_temporary_storage<host_tmp_storage<
                                    ValueType
                                    , Layout
                                    , TileI
                                    , TileJ
                                    , MinusI
                                    , MinusJ
                                    , PlusI
                                    , PlusJ
#ifndef NDEBUG
                                    , TypeTag
#endif
                                    > &>
    : boost::true_type
    {};
    
    template <typename ValueType
              , typename Layout
              , int TileI
              , int TileJ
              , int MinusI
              , int MinusJ
              , int PlusI
              , int PlusJ
#ifndef NDEBUG
              , typename TypeTag
#endif
              >
    struct is_temporary_storage<host_tmp_storage<
                                    ValueType
                                    , Layout
                                    , TileI
                                    , TileJ
                                    , MinusI
                                    , MinusJ
                                    , PlusI
                                    , PlusJ
#ifndef NDEBUG
                                    , TypeTag
#endif
                                    > const& >
    : boost::true_type
    {};
} // namespace gridtools
