#pragma once

#include <iostream>
#include <boost/type_traits/integral_constant.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/current_function.hpp>
#include "base_storage.h"
#include "hybrid_pointer.h"

#ifdef __CUDACC__

#ifdef _GT_RANDOM_INPUT
#include <stdlib.h>
#endif

//////// STORAGE

namespace gridtools {
    template < typename ValueType
             , typename Layout
             , bool IsTemporary = false
#ifndef NDEBUG
             , typename TypeTag = int
#endif
               >
    struct cuda_storage : public base_storage<cuda_storage<ValueType, Layout, IsTemporary
#ifndef NDEBUG
                                                           , TypeTag
#endif
                                                           >,
                                              ValueType,
                                              Layout,
                                              IsTemporary
                                              >
    {
        typedef cuda_storage<ValueType, Layout, IsTemporary
#ifndef NDEBUG
                             , TypeTag
#endif
                             > this_type;
        typedef base_storage<this_type, ValueType, Layout, IsTemporary> base_type;
        typedef Layout layout;
        typedef ValueType value_type;
        typedef value_type* iterator_type;
        typedef value_type const* const_iterator_type;

        hybrid_pointer<value_type> data;

        using base_type::m_dims;
        using base_type::strides;
        using base_type::m_size;
        using base_type::is_set;
        // using base_type::name;

        explicit cuda_storage(int m_dim1, int m_dim2, int m_dim3,
                value_type init = value_type(),
                std::string const& s = std::string("default name") )
        : base_type(m_dim1, m_dim2, m_dim3, init, s)
        , data(m_size)
        {
#ifdef _GT_RANDOM_INPUT
            srand(12345);
#endif
            for (int i = 0; i < m_size; ++i)
#ifdef _GT_RANDOM_INPUT
                data[i] = init * rand();
#else
                data[i] = init;
#endif
            data.update_gpu();
        }

        int name() {return 666;}

        __device__
        cuda_storage(cuda_storage const& other)
            : base_type(other)
            , data(other.data)
        { }

        explicit cuda_storage()
            : base_type()
        { }

        static void text() {
            std::cout << BOOST_CURRENT_FUNCTION << std::endl;
        }

        ~cuda_storage() { }

        /**@brief copy the pointer from the host to the device*/
        void h2d_update() {
            data.update_gpu();
        }

        /**@brief copy the pointer from the device to the host*/
        void d2h_update() {
            data.update_cpu();
        }

        GT_FUNCTION
        const_iterator_type min_addr() const {
            return &(data[0]);
        }

        GT_FUNCTION
        const_iterator_type max_addr() const {
            return &(data[m_size]);
        }

        GT_FUNCTION
        value_type& operator()(int i, int j, int k) {
            return data[base_type::_index(i,j,k)];
        }

        GT_FUNCTION
        value_type const & operator()(int i, int j, int k) const {
            return data[base_type::_index(i,j,k)];
        }

        void print() const {
            base_type::print(this);
        }

        template <typename Stream>
        void print(Stream & stream) const {
            base_type::print(this, stream);
        }
    };


    template <typename X, typename Y>
    struct is_temporary_storage<cuda_storage<X,Y,false>*& > {
        typedef boost::false_type type;
    };

#ifndef NDEBUG
    template <typename X, typename Y, typename A>
    struct is_temporary_storage<cuda_storage<X,Y,true,A>*& > {
        typedef boost::true_type type;
    };
#else
    template <typename X, typename Y>
    struct is_temporary_storage<cuda_storage<X,Y,true>*& > {
        typedef boost::true_type type;
    };
#endif

    template <typename X, typename Y>
    struct is_temporary_storage<cuda_storage<X,Y,false>* > {
        typedef boost::false_type type;
    };

#ifndef NDEBUG
    template <typename X, typename Y,typename A>
    struct is_temporary_storage<cuda_storage<X,Y,true,A>* > {
        typedef boost::true_type type;
    };
#else
    template <typename X, typename Y>
    struct is_temporary_storage<cuda_storage<X,Y,true>* > {
        typedef boost::true_type type;
    };
#endif

    template <typename X, typename Y>
    struct is_temporary_storage<cuda_storage<X,Y,false> > {
        typedef boost::false_type type;
    };

#ifndef NDEBUG
    template <typename X, typename Y, typename A>
    struct is_temporary_storage<cuda_storage<X,Y,true,A> > {
        typedef boost::true_type type;
    };
#else
    template <typename X, typename Y>
    struct is_temporary_storage<cuda_storage<X,Y,true> > {
        typedef boost::true_type type;
    };
#endif
} // namespace gridtools
#endif
