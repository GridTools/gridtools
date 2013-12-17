#pragma once

#include <cassert>
#include <iostream>
#include <boost/type_traits/integral_constant.hpp>
#include <boost/utility/enable_if.hpp>
#include "base_storage.h"
#include "basic_utils.h"
//////// STORAGE

namespace gridtools {
    template < typename t_value_type
             , typename t_layout
             , bool is_temporary = false
#ifndef NDEBUG
             , typename type_tag = int
#endif
        >
    struct storage : public base_storage<t_value_type, t_layout, is_temporary> {
        typedef base_storage<t_value_type, t_layout, is_temporary> base_type;
        typedef t_layout layout;
        typedef t_value_type value_type;
        typedef value_type* iterator_type;

        using base_type::m_dims;
        using base_type::strides;
        using base_type::m_size;
        using base_type::is_set;
        using base_type::name;

        value_type* data;

        explicit storage(int m_dim1, int m_dim2, int m_dim3,
                         value_type init = value_type(),
                         std::string const& s = std::string("default name") ) 
            : base_type(m_dim1, m_dim2, m_dim3, init, s)
        {
            data = new value_type[m_size];
            for (int i = 0; i < m_size; ++i)
                data[i] = init;
        }

        explicit storage() 
            : base_type()
        { }

        static void text() {
            std::cout << __PRETTY_FUNCTION__ << std::endl;
        }

        ~storage() {
            if (is_set) {
                std::cout << "deleting " << std::hex << data << std::endl;
                delete[] data;
            }
        }

        value_type* min_addr() const {
            return data ;
        }

        value_type* max_addr() const {
            return data+m_size;
        }

        value_type& operator()(int i, int j, int k) {
            return data[base_type::_index(i,j,k)];
        }

        value_type const & operator()(int i, int j, int k) const {
            return data[base_type::_index(i,j,k)];
        }

        void print() const {
            base_type::print(this);
        }

    };


    template <typename T>
    struct is_temporary_storage {
        typedef boost::false_type type;
    };

    template <typename X, typename Y>
    struct is_temporary_storage<storage<X,Y,false>*& > {
        typedef boost::false_type type;
    };

#ifndef NDEBUG
    template <typename X, typename Y, typename A>
    struct is_temporary_storage<storage<X,Y,true,A>*& > {
        typedef boost::true_type type;
    };
#else
    template <typename X, typename Y>
    struct is_temporary_storage<storage<X,Y,true>*& > {
        typedef boost::true_type type;
    };
#endif

    template <typename X, typename Y>
    struct is_temporary_storage<storage<X,Y,false>* > {
        typedef boost::false_type type;
    };

#ifndef NDEBUG
    template <typename X, typename Y,typename A>
    struct is_temporary_storage<storage<X,Y,true,A>* > {
        typedef boost::true_type type;
    };
#else
    template <typename X, typename Y>
    struct is_temporary_storage<storage<X,Y,true>* > {
        typedef boost::true_type type;
    };
#endif

    template <typename X, typename Y>
    struct is_temporary_storage<storage<X,Y,false> > {
        typedef boost::false_type type;
    };

#ifndef NDEBUG
    template <typename X, typename Y, typename A>
    struct is_temporary_storage<storage<X,Y,true,A> > {
        typedef boost::true_type type;
    };
#else
    template <typename X, typename Y>
    struct is_temporary_storage<storage<X,Y,true> > {
        typedef boost::true_type type;
    };
#endif
} // namespace gridtools
