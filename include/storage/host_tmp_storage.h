#pragma once

#include "storage.h"

namespace gridtools {
    template < typename ValueType
               , typename Layout
#ifndef NDEBUG
               , typename TypeTag = int
#endif
               >
    struct host_tmp_storage : public storage<ValueType,
                                             Layout,
                                             true,
#ifndef NDEBUG
                                             TypeTag
#endif
                                             > {

#ifndef NDEBUG
        typedef storage<ValueType, Layout, true, TypeTag> base_type;
#else
        typedef storage<ValueType, Layout, true> base_type;
#endif
        typedef host_tmp_storage<ValueType, Layout> this_type;

        typedef Layout layout;
        typedef ValueType value_type;
        typedef value_type* iterator_type;
        typedef value_type const* const_iterator_type;

        using base_type::m_dims;
        using base_type::strides;
        using base_type::m_size;
        using base_type::is_set;
        using base_type::m_name;
        const int offs_i;
        const int offs_j;
        const int offs_k;

        explicit host_tmp_storage(int m_dim1, int m_dim2, int m_dim3,
                                  int offs_i, int offs_j, int offs_k,
                                  value_type init = value_type(),
                                  std::string const& s = std::string("default name") )
            : base_type(m_dim1, m_dim2, m_dim3, init, s)
            , offs_i(offs_i)
            , offs_j(offs_j)
            , offs_k(offs_k)
        {}

        host_tmp_storage()
        : offs_i()
        , offs_j()
        , offs_k()
        {}

        virtual int offset(int i, int j, int k) const {
            return layout::template find<2>(m_dims) * layout::template find<1>(m_dims)
            * layout::template find<0>(i-offs_i,j-offs_j,k-offs_k) +
            layout::template find<2>(m_dims) * layout::template find<1>(i-offs_i,j-offs_j,k-offs_k) +
            layout::template find<2>(i-offs_i,j-offs_j,k-offs_k);
        }

        virtual void info() const {
            std::cout << "Temporary storage "
                      << m_dims[0] << "x"
                      << m_dims[1] << "x"
                      << m_dims[2] << ", "
                      << m_name
                      << std::endl;
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

    template <typename T, typename U>
    std::ostream& operator<<(std::ostream &s, host_tmp_storage<T,U> ) {
        return s << "host_tmp_storage <T,U>" ;
    }

} // namespace gridtools
