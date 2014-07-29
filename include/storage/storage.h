#pragma once

#include <iostream>
#include <boost/type_traits/integral_constant.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/current_function.hpp>
#include "base_storage.h"
//////// STORAGE

namespace gridtools {
    template < typename ValueType
             , typename Layout
             , bool IsTemporary = false
#ifndef NDEBUG
             , typename TypeTag = int
#endif
        >
    struct storage : public base_storage<storage<ValueType, Layout, IsTemporary>,
            ValueType,
            Layout,
            IsTemporary> {
        typedef storage<ValueType, Layout, IsTemporary> this_type;
        typedef base_storage<this_type, ValueType, Layout, IsTemporary> base_type;
        typedef Layout layout;
        typedef ValueType value_type;
        typedef value_type* iterator_type;
        typedef value_type const* const_iterator_type;

        using base_type::m_dims;
        using base_type::strides;
        using base_type::m_size;
        using base_type::is_set;
        std::string m_name;

        value_type* data;

        explicit storage(int m_dim1, int m_dim2, int m_dim3,
                         value_type init = value_type(),
                         std::string const& s = std::string("default name") ) 
            : base_type(m_dim1, m_dim2, m_dim3, init, s)
            , m_name(s)
        {
            data = new value_type[m_size];
            for (int i = 0; i < m_size; ++i)
                data[i] = init;
        }

        std::string const& name() const {
            return m_name;
        }

        explicit storage() 
            : base_type()
        { }

        static void text() {
            std::cout << BOOST_CURRENT_FUNCTION << std::endl;
        }

        virtual void info() const {
            std::cout << m_dims[0] << "x"
                      << m_dims[1] << "x"
                      << m_dims[2] << ", "
                      << m_name
                      << std::endl;
        }

        ~storage() {
            if (is_set) {
                //std::cout << "deleting " << std::hex << data << std::endl;
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


    template <typename X, typename Y>
    struct is_temporary_storage<storage<X,Y,false>*& >
      : boost::false_type
    {};

#ifndef NDEBUG
    template <typename X, typename Y, typename A>
    struct is_temporary_storage<storage<X,Y,true,A>*& >
      : boost::true_type
    {};
#else
    template <typename X, typename Y>
    struct is_temporary_storage<storage<X,Y,true>*& >
      : boost::true_type
    {};
#endif

    template <typename X, typename Y>
    struct is_temporary_storage<storage<X,Y,false>* >
      : boost::false_type
    {};

#ifndef NDEBUG
    template <typename X, typename Y,typename A>
    struct is_temporary_storage<storage<X,Y,true,A>* >
      : boost::true_type
    {};
#else
    template <typename X, typename Y>
    struct is_temporary_storage<storage<X,Y,true>* >
      : boost::true_type
    {};
#endif

    template <typename X, typename Y>
    struct is_temporary_storage<storage<X,Y,false> >
      : boost::false_type
    {};

#ifndef NDEBUG
    template <typename X, typename Y, typename A>
    struct is_temporary_storage<storage<X,Y,true,A> >
      : boost::true_type
    {};
#else
    template <typename X, typename Y>
    struct is_temporary_storage<storage<X,Y,true> >
      : boost::true_type
    {};
#endif
} // namespace gridtools
