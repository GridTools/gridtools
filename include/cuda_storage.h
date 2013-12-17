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
    struct cuda_storage {
        typedef t_layout layout;
        typedef t_value_type value_type;
        typedef value_type* iterator_type;

        int m_dims[3];
        int strides[3];
        int m_size;
        value_type* host_data;
        value_type* acc_data;
        bool is_set;
        std::string name;

        explicit cuda_storage(int m_dim1, int m_dim2, int m_dim3,
                         value_type init = value_type(),
                         std::string const& s = std::string("default name") ) {
            m_dims[0] = m_dim1;
            m_dims[1] = m_dim2;
            m_dims[2] = m_dim3;
            strides[0] = layout::template find<2>(m_dims)*layout::template find<1>(m_dims);
            strides[1] = layout::template find<2>(m_dims);
            strides[2] = 1;
            m_size = m_dims[0] * m_dims[1] * m_dims[2];
            std::cout << "Size " << m_size << std::endl;
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
            for (int i = 0; i < m_size; ++i)
                host_data[i] = init;
            is_set=true;
            name = s;
        }

        explicit cuda_storage() {is_set=false;}

        static void text() {
            std::cout << __PRETTY_FUNCTION__ << std::endl;
        }

        void info() const {
            text();
            std::cout << m_dims[0] << "x"
                      << m_dims[1] << "x"
                      << m_dims[2] << ", "
                      << name << std::endl;
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
            return acc_data[_index(i,j,k)];
#else
            return host_data[_index(i,j,k)];
#endif
        }

        __host__ __device__
        value_type const & operator()(int i, int j, int k) const {
#ifdef __CUDA_ARCH__
            return acc_data[_index(i,j,k)];
#else
            return host_data[_index(i,j,k)];
#endif
        }

        void print() const {
            std::cout << "Printing " << name << std::endl;
            std::cout << "(" << m_dims[0] << "x"
                      << m_dims[1] << "x"
                      << m_dims[2] << ")"
                      << std::endl;
            std::cout << "| j" << std::endl;
            std::cout << "| j" << std::endl;
            std::cout << "v j" << std::endl;
            std::cout << "---> k" << std::endl;

            for (int i = 0; i < std::min(m_dims[0],6); ++i) {
                for (int j = 0; j < std::min(m_dims[1],6); ++j) {
                    for (int k = 0; k < std::min(m_dims[2],12); ++k) {
                        std::cout << "["/*("
                                          << i << ","
                                          << j << ","
                                          << k << ")"*/
                                  << operator()(i,j,k) << "] ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        template <int I>
        int stride_along() const {
            return get_stride<I, layout>::get(strides); /*layout::template at_<I>::value];*/
        }

        template <typename t_offset>
        int compute_offset(t_offset const& offset) const {
            return layout::template find<2>(m_dims) * layout::template find<1>(m_dims)
                * layout::template find<0>(offset.offset_ptr()) +
                layout::template find<2>(m_dims) * layout::template find<1>(offset.offset_ptr()) +
                layout::template find<2>(offset.offset_ptr());
        }

    private:
        template <typename t_dummy, int X>
        struct _is_0: public boost::false_type
        {};

        template <typename t_dummy>
        struct _is_0<t_dummy,0>: public boost::true_type
        { };

        template <typename t_dummy, int X>
        struct _is_2: public boost::false_type
        {};

        template <typename t_dummy>
        struct _is_2<t_dummy,2>: public boost::true_type
        { };

        template <int I, typename _t_layout, typename ENABLE=void>
        struct get_stride;

        template <int I, typename _t_layout>
        struct get_stride<I, _t_layout, typename boost::enable_if<
                                            _is_2< void, _t_layout::template at_<I>::value >
                                            >::type> {
            static int get(const int* ) {
#ifndef NDEBUG
                std::cout << "U" ;//<< std::endl;
#endif
                return 1;
            }
        };

template <int I, typename _t_layout>
struct get_stride<I, _t_layout, typename boost::disable_if<
                                    _is_2<void, _t_layout::template at_<I>::value>
                                    >::type> {
    static int get(const int* s) {
        return s[_t_layout::template at_<I>::value];
    }
};

        __host__ __device__
        int _index(int i, int j, int k) const {
            int index;
            if (is_temporary) {
                index =
                    layout::template find<2>(m_dims) * layout::template find<1>(m_dims)
                    * (modulus(layout::template find<0>(i,j,k),layout::template find<0>(m_dims))) +
                    layout::template find<2>(m_dims) * modulus(layout::template find<1>(i,j,k),layout::template find<1>(m_dims)) +
                    modulus(layout::template find<2>(i,j,k),layout::template find<2>(m_dims));
            } else {
                index =
                    layout::template find<2>(m_dims) * layout::template find<1>(m_dims)
                    * layout::template find<0>(i,j,k) +
                    layout::template find<2>(m_dims) * layout::template find<1>(i,j,k) +
                    layout::template find<2>(i,j,k);
            }
            assert(index >= 0);
            assert(index <m_size);
            return index;
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
