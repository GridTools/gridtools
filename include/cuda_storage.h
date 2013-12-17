#pragma once

#include <cassert>
#include <iostream>
#include <boost/type_traits/integral_constant.hpp>
#include <boost/utility/enable_if.hpp>
#include "basic_utils.h"
#ifdef __CUDACC__

//////// STORAGE

namespace gridtools {
    template < typename t_value_type
             , typename t_layout
             , bool is_temporary = false
#ifndef NDEBUG
             , typename type_tag = int
#endif
        >
    struct cuda_storage : public base_storage<t_value_type, t_layout, is_temporary> {
        typedef base_storage<t_value_type, t_layout, is_temporary> base_type;
        typedef t_layout layout;
        typedef t_value_type value_type;
        typedef value_type* iterator_type;

        using base_type::m_dims;
        using base_type::strides;
        using base_type::m_size;
        using base_type::is_set;
        using base_type::name;

        value_type* host_data;
        value_type* acc_data;

        explicit cuda_storage(int m_dim1, int m_dim2, int m_dim3,
                         value_type init = value_type(),
                              std::string const& s = std::string("default name") )
            : base_type(m_dim1, m_dim2, m_dim3, init, s)
        {
            host_data = new value_type[m_size];
            int err = cudaMalloc(&acc_data, m_size*sizeof(value_type));
            if (err != cudaSuccess) {
                std::cout << "Error allocating storage in "
                          << __PRETTY_FUNCTION__
                          << " : size = "
                          << m_size*sizeof(value_type)
                          << " bytes "
                          << std::endl;
            }
        }

        explicit cuda_storage()
            : base_type()
        { }

        static void text() {
            std::cout << __PRETTY_FUNCTION__ << std::endl;
        }

        ~cuda_storage() {
            if (is_set) {
                std::cout << "deleting " << std::hex << host_data << std::endl;
                delete[] host_data;
                cudaFree(acc_data);
            }
        }

        value_type* min_addr() const {
#ifdef __CUDA_ARCH__
            return acc_data ;
#else
            return host_data ;
#endif
        }

        value_type* max_addr() const {
#ifdef __CUDA_ARCH__
            return acc_data+m_size;
#else
            return host_data+m_size;
#endif
        }

        __host__ __device__
        value_type& operator()(int i, int j, int k) {
#ifdef __CUDA_ARCH__
            return acc_data[base_type::_index(i,j,k)];
#else
            return host_data[base_type::_index(i,j,k)];
#endif
        }

        __host__ __device__
        value_type const & operator()(int i, int j, int k) const {
#ifdef __CUDA_ARCH__
            return acc_data[base_type::_index(i,j,k)];
#else
            return host_data[base_type::_index(i,j,k)];
#endif
        }

        void print() const {
            base_type::print(this);
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
