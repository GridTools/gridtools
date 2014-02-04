#pragma once
#include "basic_utils.h"
#include "gpu_clone.h"

namespace gridtools {

    template < typename derived,
               typename t_value_type,
               typename t_layout,
               bool is_temporary = false
               >
    struct base_storage : public clonable_to_gpu<derived> {
        typedef t_layout layout;
        typedef t_value_type value_type;
        typedef value_type* iterator_type;

        int m_dims[3];
        int strides[3];
        int m_size;
        bool is_set;
        std::string name;

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
            std::cout << "Size " << m_size << std::endl;
            is_set=true;
            name = s;
        }

        explicit base_storage() {
            is_set=false;
        }

        virtual void h2d_update() const {}
        virtual void d2h_update() const {}

        void info() const {
            std::cout << m_dims[0] << "x"
                      << m_dims[1] << "x"
                      << m_dims[2] << ", "
                      << name << std::endl;
        }

        template <int I>
        GT_FUNCTION
        int stride_along() const {
            return get_stride<I, layout>::get(strides); /*layout::template at_<I>::value];*/
        }

        template <typename t_offset>
        GT_FUNCTION
        int compute_offset(t_offset const& offset) const {
            return layout::template find<2>(m_dims) * layout::template find<1>(m_dims)
                * layout::template find<0>(offset.offset_ptr()) +
                layout::template find<2>(m_dims) * layout::template find<1>(offset.offset_ptr()) +
                layout::template find<2>(offset.offset_ptr());
        }

    protected:
        template <typename derived_t>
        void print(derived_t* that) const {
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
                                  << that->operator()(i,j,k) << "] ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

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
            GT_FUNCTION
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
            GT_FUNCTION
            static int get(const int* s) {
                return s[_t_layout::template at_<I>::value];
            }
        };

        GT_FUNCTION
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

}
