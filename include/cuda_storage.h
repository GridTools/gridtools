#pragma once

#include <iostream>
#include <boost/type_traits/integral_constant.hpp>
#include <boost/utility/enable_if.hpp>
#include "base_storage.h"
#include "hybrid_pointer.h"

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
    struct cuda_storage : public base_storage<cuda_storage<t_value_type, t_layout, is_temporary>,
            t_value_type,
            t_layout,
            is_temporary> {
        typedef cuda_storage<t_value_type, t_layout, is_temporary> this_type;
        typedef base_storage<this_type, t_value_type, t_layout, is_temporary> base_type;
        typedef t_layout layout;
        typedef t_value_type value_type;
        typedef value_type* iterator_type;

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
            data.update_gpu();
        }

        explicit cuda_storage()
            : base_type()
        { }

        static void text() {
            std::cout << __PRETTY_FUNCTION__ << std::endl;
        }

        ~cuda_storage() { }

        void h2d_update() {
            data.update_gpu();
        }

        void d2h_update() {
            data.update_cpu();
        }

        GT_FUNCTION
        value_type const* min_addr() const {
            return &(data[0]);
        }

        GT_FUNCTION
        value_type const * max_addr() const {
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
