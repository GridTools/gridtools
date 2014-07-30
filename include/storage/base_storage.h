#pragma once
#include "../common/basic_utils.h"
#include "../common/gpu_clone.h"
#include "../common/gt_assert.h"

namespace gridtools {

    namespace _impl
    {
        template <int I, typename OtherLayout, int X>
        struct get_stride_
        {
            GT_FUNCTION
            static int get(const int* s) {
                return s[OtherLayout::template at_<I>::value];
            }
        };

        template <int I, typename OtherLayout>
        struct get_stride_<I, OtherLayout, 2>
        {
            GT_FUNCTION
            static int get(const int* ) {
#ifndef __CUDACC__
#ifndef NDEBUG
                //                std::cout << "U" ;//<< std::endl;
#endif
#endif
                return 1;
            }
        };

        template <int I, typename OtherLayout>
        struct get_stride
          : get_stride_<I, OtherLayout, OtherLayout::template at_<I>::value>
        {};
    }

    template < typename Derived,
               typename ValueType,
               typename Layout,
               bool IsTemporary = false
               >
    struct base_storage : public clonable_to_gpu<Derived> {
        typedef Layout layout;
        typedef ValueType value_type;
        typedef value_type* iterator_type;
        typedef value_type const* const_iterator_type;

        int m_dims[3];
        int strides[3];
        int m_size;
        bool is_set;
        //std::string name;

        explicit base_storage(int m_dim1, int m_dim2, int m_dim3,
                         value_type init = value_type(),
                         std::string const& s = std::string("default name") ) {
            m_dims[0] = m_dim1;
            m_dims[1] = m_dim2;
            m_dims[2] = m_dim3;
            strides[0] = layout::template find<2>(m_dims)*layout::template find<1>(m_dims);
            strides[1] = layout::template find<2>(m_dims);
            strides[2] = 1;
            m_size = m_dims[0] * m_dims[1] * m_dims[2];
            //            std::cout << "Size " << m_size << std::endl;
            is_set=true;
            //name = s;
        }

        __device__
        base_storage(base_storage const& other)
            : m_size(other.m_size)
            , is_set(is_set)
        {
            m_dims[0] = other.m_dims[0];
            m_dims[1] = other.m_dims[1];
            m_dims[2] = other.m_dims[2];

            strides[0] = other.strides[0];
            strides[1] = other.strides[1];
            strides[2] = other.strides[2];
        }

        explicit base_storage() {
            is_set=false;
        }

        virtual void h2d_update() {}
        virtual void d2h_update() {}

        virtual void info() const {
            std::cout << m_dims[0] << "x"
                      << m_dims[1] << "x"
                      << m_dims[2] << ", "
                      << std::endl;
        }

        template <int I>
        GT_FUNCTION
        int stride_along() const {
            return _impl::get_stride<I, layout>::get(strides); /*layout::template at_<I>::value];*/
        }

        int offset(int i, int j, int k) const {
            return _index(i,j,k);
        }

    protected:
        template <typename derived_t>
        void print(derived_t* that) const {
            print(that, std::cout);
        }

        template <typename derived_t, typename Stream>
        void print(derived_t* that, Stream & stream) const {
            //std::cout << "Printing " << name << std::endl;
            stream << "(" << m_dims[0] << "x"
                      << m_dims[1] << "x"
                      << m_dims[2] << ")"
                      << std::endl;
            stream << "| j" << std::endl;
            stream << "| j" << std::endl;
            stream << "v j" << std::endl;
            stream << "---> k" << std::endl;

            int MI=12;
            int MJ=12;
            int MK=12;

            for (int i = 0; i < m_dims[0]; i += std::max(1,m_dims[0]/MI)) {
                for (int j = 0; j < m_dims[1]; j += std::max(1,m_dims[1]/MJ)) {
                    for (int k = 0; k < m_dims[2]; k += std::max(1,m_dims[2]/MK)) {
                        stream << "["/*("
                                          << i << ","
                                          << j << ","
                                          << k << ")"*/
                                  << that->operator()(i,j,k) << "] ";
                    }
                    stream << std::endl;
                }
                stream << std::endl;
            }
            stream << std::endl;
        }

        GT_FUNCTION
        int _index(int i, int j, int k) const {
            int index;
            if (IsTemporary) {
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
            //assert(index >= 0);
            assert(index <m_size);
            return index;
        }
    };
    
    template <typename T>
    struct is_temporary_storage {
        typedef boost::false_type type;
    };



} //namespace gridtools
