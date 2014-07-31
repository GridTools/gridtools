#pragma once

#include "base_storage.h"

namespace gridtools {
    template < typename ValueType
               , typename Layout
#ifndef NDEBUG
               , typename TypeTag = int
#endif
               >
    struct host_tmp_storage : public base_storage<host_tmp_storage<
                                                      ValueType
                                                      , Layout
#ifndef NDEBUG
                                                      , TypeTag
#endif
                                                      >,
                                                  ValueType,
                                                  Layout,
                                                  true
                                                  > {

        typedef base_storage<host_tmp_storage<
                                 ValueType
                                 , Layout
#ifndef NDEBUG
                                 , TypeTag
#endif
                                 >
                             ,ValueType
                             , Layout
                             , true> base_type;

#ifndef NDEBUG
        typedef host_tmp_storage<ValueType, Layout, TypeTag> this_type;
#else
        typedef host_tmp_storage<ValueType, Layout> this_type;
#endif
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

        explicit host_tmp_storage(int m_dim1, int m_dim2, int m_dim3,
                                  int offs_i_m, int offs_j_m, int offs_k_m,
                                  int offs_i_p, int offs_j_p, int offs_k_p,
                                  value_type init = value_type(),
                                  std::string const& s = std::string("default name") )
            : base_type(m_dim1+offs_i_m+offs_i_p, m_dim2+offs_j_m+offs_j_p, m_dim3+offs_k_m+offs_k_p, init)
            , m_name(s)
        {
            m_tile[0] = m_dim1;
            m_tile[1] = m_dim2;
            m_tile[2] = m_dim3;
            offs[0]=offs_i_m;
            offs[1]=offs_j_m;
            offs[2]=offs_k_m;
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

        value_type& operator()(int i, int j, int k) {
            assert(_index(i,j,k) >= 0);
            assert(_index(i,j,k) < m_size);
            return data[_index(i,j,k)];
        }

        value_type const & operator()(int i, int j, int k) const {
            assert(_index(i,j,k) >= 0);
            assert(_index(i,j,k) < m_size);
            return data[_index(i,j,k)];
        }

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
        
#ifndef NDEBUG
    template <typename X, typename Y, typename A>
    struct is_temporary_storage<host_tmp_storage<X,Y,A>*& >
      : boost::true_type
    {};
#else
    template <typename X, typename Y>
    struct is_temporary_storage<host_tmp_storage<X,Y>*& >
      : boost::true_type
    {};
#endif

#ifndef NDEBUG
    template <typename X, typename Y,typename A>
    struct is_temporary_storage<host_tmp_storage<X,Y,A>* >
      : boost::true_type
    {};
#else
    template <typename X, typename Y>
    struct is_temporary_storage<host_tmp_storage<X,Y>* >
      : boost::true_type
    {};
#endif

#ifndef NDEBUG
    template <typename X, typename Y, typename A>
    struct is_temporary_storage<host_tmp_storage<X,Y,A> >
      : boost::true_type
    {};
#else
    template <typename X, typename Y>
    struct is_temporary_storage<host_tmp_storage<X,Y> >
      : boost::true_type
    {};
#endif

#ifndef NDEBUG
    template <typename T, typename U, typename TT>
    std::ostream& operator<<(std::ostream &s, host_tmp_storage<T,U,TT> ) {
        return s << "host_tmp_storage <T,U,TT>" ;
    }
#else
    template <typename T, typename U>
    std::ostream& operator<<(std::ostream &s, host_tmp_storage<T,U> ) {
        return s << "host_tmp_storage <T,U>" ;
    }
#endif

} // namespace gridtools
