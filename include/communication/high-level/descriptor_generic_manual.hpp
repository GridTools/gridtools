/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#ifndef _DESCRIPTOR_GENERIC_MANUAL_H_
#define _DESCRIPTOR_GENERIC_MANUAL_H_

#include "communication/high-level/gcl_parameters.hpp"
#include "communication/high-level/descriptor_base.hpp"

#ifdef __CUDACC__
#include "communication/high-level/m_packZL_generic.hpp"
#include "communication/high-level/m_packZU_generic.hpp"
#include "communication/high-level/m_packYL_generic.hpp"
#include "communication/high-level/m_packYU_generic.hpp"
#include "communication/high-level/m_packXL_generic.hpp"
#include "communication/high-level/m_packXU_generic.hpp"

#include "communication/high-level/m_unpackZL_generic.hpp"
#include "communication/high-level/m_unpackZU_generic.hpp"
#include "communication/high-level/m_unpackYL_generic.hpp"
#include "communication/high-level/m_unpackYU_generic.hpp"
#include "communication/high-level/m_unpackXL_generic.hpp"
#include "communication/high-level/m_unpackXU_generic.hpp"

#define KERNEL_TYPE ZL
#include "communication/high-level/call_generic.hpp"
#undef KERNEL_TYPE

#define KERNEL_TYPE ZU
#include "communication/high-level/call_generic.hpp"
#undef KERNEL_TYPE

#define KERNEL_TYPE YL
#include "communication/high-level/call_generic.hpp"
#undef KERNEL_TYPE

#define KERNEL_TYPE YU
#include "communication/high-level/call_generic.hpp"
#undef KERNEL_TYPE

#define KERNEL_TYPE XL
#include "communication/high-level/call_generic.hpp"
#undef KERNEL_TYPE

#define KERNEL_TYPE XU
#include "communication/high-level/call_generic.hpp"
#undef KERNEL_TYPE
#endif

namespace gridtools {

    template < typename HaloExch, typename proc_layout_abs >
    class hndlr_generic< 3, HaloExch, proc_layout_abs, gcl_cpu, version_manual > : public descriptor_base< HaloExch > {
        static const int DIMS = 3;
        gridtools::array< char *, _impl::static_pow3< DIMS >::value > send_buffer; // One entry will not be used...
        gridtools::array< char *, _impl::static_pow3< DIMS >::value > recv_buffer;
        gridtools::array< int, _impl::static_pow3< DIMS >::value > send_buffer_size; // One entry will not be used...
        gridtools::array< int, _impl::static_pow3< DIMS >::value > recv_buffer_size;

      public:
        typedef descriptor_base< HaloExch > base_type;
        typedef typename base_type::pattern_type pattern_type;
        /** Architecture type
           */
        typedef gcl_cpu arch_type;

        /**
           Type of the computin grid associated to the pattern
         */
        typedef typename base_type::grid_type grid_type;

        /**
           Type of the translation used to map dimensions to buffer addresses
         */
        typedef translate_t< DIMS, typename default_layout_map< DIMS >::type > translate;

        hndlr_generic(grid_type const &g) : base_type(g) {}

        ~hndlr_generic() {
#ifdef _GCL_CHECK_DESTRUCTOR
            std::cout << "Destructor " << __FILE__ << ":" << __LINE__ << std::endl;
#endif

            for (int i = -1; i <= 1; ++i)
                for (int j = -1; j <= 1; ++j)
                    for (int k = -1; k <= 1; ++k) {
                        _impl::gcl_alloc< char, arch_type >::free(send_buffer[translate()(i, j, k)]);
                        _impl::gcl_alloc< char, arch_type >::free(recv_buffer[translate()(i, j, k)]);
                    }
        }

        /**
           Setup function, in this version, takes tree parameters to
           compute internal buffers and sizes. It takes a field on the fly
           struct, which requires Datatype and layout map template
           arguments that are inferred, so the user is not aware of them.

           \tparam DataType This type is inferred by halo_example paramter
           \tparam t_layoutmap This type is inferred by halo_example paramter

           \param[in] max_fields_n Maximum number of grids used in a computation
           \param[in] halo_example The (at least) maximal grid that is goinf to be used
           \param[in] typesize In case the DataType of the halo_example is not the same as the maximum data type used in
           the computation, this parameter can be given
         */
        template < typename DataType, typename f_layoutmap, template < typename > class traits >
        void setup(int max_fields_n,
            field_on_the_fly< DataType, f_layoutmap, traits > const &halo_example,
            int typesize = sizeof(DataType)) {

            typedef typename field_on_the_fly< DataType, f_layoutmap, traits >::inner_layoutmap t_layoutmap;
            gridtools::array< int, DIMS > eta;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    for (int k = -1; k <= 1; ++k) {
                        if (i != 0 || j != 0 || k != 0) {
                            eta[0] = i;
                            eta[1] = j;
                            eta[2] = k;
                            int S = 1;
                            S = halo_example.send_buffer_size(eta);
                            int R = 1;
                            R = halo_example.recv_buffer_size(eta);

                            send_buffer_size[translate()(i, j, k)] = (S * max_fields_n * typesize);
                            recv_buffer_size[translate()(i, j, k)] = (R * max_fields_n * typesize);

                            // std::cout << halo_example << std::endl;
                            // std::cout << "Send size to "
                            //           << i << ", "
                            //           << j << ", "
                            //           << k << ": "
                            //           << send_buffer_size[translate()(i,j,k)]
                            //           << std::endl;
                            // std::cout << "Recv size fr "
                            //           << i << ", "
                            //           << j << ", "
                            //           << k << ": "
                            //           << recv_buffer_size[translate()(i,j,k)]
                            //           << std::endl;
                            // std::cout << std::endl;
                            // std::cout.flush();

                            send_buffer[translate()(i, j, k)] =
                                _impl::gcl_alloc< char, arch_type >::alloc(send_buffer_size[translate()(i, j, k)]);
                            recv_buffer[translate()(i, j, k)] =
                                _impl::gcl_alloc< char, arch_type >::alloc(recv_buffer_size[translate()(i, j, k)]);

                            typedef typename layout_transform< t_layoutmap, proc_layout_abs >::type proc_layout;
                            const int i_P = pack_get_elem< proc_layout::template at< 0 >() >::apply(i, j, k);
                            const int j_P = pack_get_elem< proc_layout::template at< 1 >() >::apply(i, j, k);
                            const int k_P = pack_get_elem< proc_layout::template at< 2 >() >::apply(i, j, k);

                            base_type::m_haloexch.register_send_to_buffer(&(send_buffer[translate()(i, j, k)][0]),
                                send_buffer_size[translate()(i, j, k)],
                                i_P,
                                j_P,
                                k_P);

                            base_type::m_haloexch.register_receive_from_buffer(&(recv_buffer[translate()(i, j, k)][0]),
                                recv_buffer_size[translate()(i, j, k)],
                                i_P,
                                j_P,
                                k_P);
                        }
                    }
                }
            }
        }

        /**
           Setup function, in this version, takes a single parameter with
           an array of sizes to be associated with the halos.

           \tparam DataType This type is inferred by halo_example paramter
           \tparam t_layoutmap This type is inferred by halo_example paramter

           \param[in] buffer_size_list Array (gridtools::array) with the sizes of the buffers associated with the halos.
         */
        template < typename DataType, typename t_layoutmap >
        void setup(gridtools::array< size_t, _impl::static_pow3< DIMS >::value > const &buffer_size_list) {
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    for (int k = -1; k <= 1; ++k) {
                        if (i != 0 || j != 0 || k != 0) {
                            send_buffer[translate()(i, j, k)] =
                                _impl::gcl_alloc< char, arch_type >::alloc(buffer_size_list[translate()(i, j, k)]);
                            recv_buffer[translate()(i, j, k)] =
                                _impl::gcl_alloc< char, arch_type >::alloc(buffer_size_list[translate()(i, j, k)]);
                            send_buffer_size[translate()(i, j, k)] = (buffer_size_list[translate()(i, j, k)]);
                            recv_buffer_size[translate()(i, j, k)] = (buffer_size_list[translate()(i, j, k)]);

                            typedef typename layout_transform< t_layoutmap, proc_layout_abs >::type proc_layout;
                            const int i_P = pack_get_elem< proc_layout::template at< 0 >() >::apply(i, j, k);
                            const int j_P = pack_get_elem< proc_layout::template at< 1 >() >::apply(i, j, k);
                            const int k_P = pack_get_elem< proc_layout::template at< 2 >() >::apply(i, j, k);

                            base_type::m_haloexch.register_send_to_buffer(&(send_buffer[translate()(i, j, k)][0]),
                                buffer_size_list[translate()(i, j, k)],
                                i_P,
                                j_P,
                                k_P);

                            base_type::m_haloexch.register_receive_from_buffer(&(recv_buffer[translate()(i, j, k)][0]),
                                buffer_size_list[translate()(i, j, k)],
                                i_P,
                                j_P,
                                k_P);
                        }
                    }
                }
            }
        }

#ifdef CXX11_ENABLED
        template < typename... FIELDS >
        void pack(const FIELDS &... _fields) const {
            for (int ii = -1; ii <= 1; ++ii) {
                for (int jj = -1; jj <= 1; ++jj) {
                    for (int kk = -1; kk <= 1; ++kk) {
                        char *it = reinterpret_cast< char * >(&(send_buffer[translate()(ii, jj, kk)][0]));
                        pack_dims< DIMS, 0 >()(*this, /*make_array(*/ ii, jj, kk /*)*/, it, _fields...);
                    }
                }
            }
        }
//}
#else
#define MACRO_IMPL(z, n, _)                                                                                            \
    template < BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(n), typename FIELD) >                                            \
    void pack(BOOST_PP_ENUM_BINARY_PARAMS_Z(z, BOOST_PP_INC(n), FIELD, const &_field)) const {                         \
        for (int ii = -1; ii <= 1; ++ii) {                                                                             \
            for (int jj = -1; jj <= 1; ++jj) {                                                                         \
                for (int kk = -1; kk <= 1; ++kk) {                                                                     \
                    char *it = reinterpret_cast< char * >(&(send_buffer[translate()(ii, jj, kk)][0]));                 \
                    pack_dims< DIMS, 0 >()(*this, ii, jj, kk, it, BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(n), _field)); \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

        BOOST_PP_REPEAT(GCL_MAX_FIELDS, MACRO_IMPL, all)
#undef MACRO_IMPL
#endif

#ifdef CXX11_ENABLED
        template < typename... FIELDS >
        void unpack(const FIELDS &... _fields) const {
            for (int ii = -1; ii <= 1; ++ii) {
                for (int jj = -1; jj <= 1; ++jj) {
                    for (int kk = -1; kk <= 1; ++kk) {
                        char *it = reinterpret_cast< char * >(&(recv_buffer[translate()(ii, jj, kk)][0]));
                        unpack_dims< DIMS, 0 >()(*this, ii, jj, kk, it, _fields...);
                    }
                }
            }
        }

#else
#define MACRO_IMPL(z, n, _)                                                                            \
    template < BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(n), typename FIELD) >                            \
    void unpack(BOOST_PP_ENUM_BINARY_PARAMS_Z(z, BOOST_PP_INC(n), FIELD, const &_field)) const {       \
        for (int ii = -1; ii <= 1; ++ii) {                                                             \
            for (int jj = -1; jj <= 1; ++jj) {                                                         \
                for (int kk = -1; kk <= 1; ++kk) {                                                     \
                    char *it = reinterpret_cast< char * >(&(recv_buffer[translate()(ii, jj, kk)][0])); \
                    unpack_dims< DIMS, 0 >()(                                                          \
                        *this, ii, jj, kk, it, BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(n), _field));    \
                }                                                                                      \
            }                                                                                          \
        }                                                                                              \
    }

        BOOST_PP_REPEAT(GCL_MAX_FIELDS, MACRO_IMPL, all)
#undef MACRO_IMPL
#endif

        /**
           Function to unpack received data

           \tparam array_of_fotf this should be an array of field_on_the_fly
           \param[in] fields vector with fields on the fly
        */
        template < typename T1, typename T2, template < typename > class T3 >
        void pack(std::vector< field_on_the_fly< T1, T2, T3 > > const &fields) {
            for (int ii = -1; ii <= 1; ++ii) {
                for (int jj = -1; jj <= 1; ++jj) {
                    for (int kk = -1; kk <= 1; ++kk) {
                        typename field_on_the_fly< T1, T2, T3 >::value_type *it =
                            reinterpret_cast< typename field_on_the_fly< T1, T2, T3 >::value_type * >(
                                &(send_buffer[translate()(ii, jj, kk)][0]));
                        pack_vector_dims< DIMS, 0 >()(*this, ii, jj, kk, it, fields);
                    }
                }
            }
        }

        /**
           Function to unpack received data

           \tparam array_of_fotf this should be an array of field_on_the_fly
           \param[in] fields vector with fields on the fly
        */
        template < typename T1, typename T2, template < typename > class T3 >
        void unpack(std::vector< field_on_the_fly< T1, T2, T3 > > const &fields) {
            for (int ii = -1; ii <= 1; ++ii) {
                for (int jj = -1; jj <= 1; ++jj) {
                    for (int kk = -1; kk <= 1; ++kk) {
                        typename field_on_the_fly< T1, T2, T3 >::value_type *it =
                            reinterpret_cast< typename field_on_the_fly< T1, T2, T3 >::value_type * >(
                                &(recv_buffer[translate()(ii, jj, kk)][0]));
                        unpack_vector_dims< DIMS, 0 >()(*this, ii, jj, kk, it, fields);
                    }
                }
            }
        }

      private:
        template < int, int >
        struct pack_dims {};

        template < int dummy >
        struct pack_dims< 3, dummy > {

            template < typename T, typename iterator >
            void operator()(const T &, int, int, int, iterator &) const {}

#ifdef CXX11_ENABLED
            template < typename T, typename iterator, typename FIRST, typename... FIELDS >
            void operator()(
                const T &hm, int ii, int jj, int kk, iterator &it, FIRST const &first, const FIELDS &... _fields)
                const {
                typedef typename layout_transform< typename FIRST::inner_layoutmap, proc_layout_abs >::type proc_layout;
                const int ii_P = pack_get_elem< proc_layout::template at< 0 >() >::apply(ii, jj, kk);
                const int jj_P = pack_get_elem< proc_layout::template at< 1 >() >::apply(ii, jj, kk);
                const int kk_P = pack_get_elem< proc_layout::template at< 2 >() >::apply(ii, jj, kk);
                if ((ii != 0 || jj != 0 || kk != 0) && (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                    first.pack(make_array(ii, jj, kk), first.ptr, it);
                    operator()(hm, ii, jj, kk, it, _fields...);
                }
            }
#else
//#define MBUILD(n) _field ## n
#define _CALLNEXT_INST(z, m, n) , _field##m
#define CALLNEXT_INST(m) BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(m), _CALLNEXT_INST, m)

#define MACRO_IMPL(z, n, _)                                                                                       \
    template < typename T, typename iterator, BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(n), typename FIELD) >        \
    void operator()(const T &hm,                                                                                  \
        int ii,                                                                                                   \
        int jj,                                                                                                   \
        int kk,                                                                                                   \
        iterator &it,                                                                                             \
        BOOST_PP_ENUM_BINARY_PARAMS_Z(z, BOOST_PP_INC(n), FIELD, const &_field)) const {                          \
        typedef typename layout_transform< typename FIELD0::inner_layoutmap, proc_layout_abs >::type proc_layout; \
        const int ii_P = pack_get_elem< proc_layout::template at< 0 >() >::apply(ii, jj, kk);                     \
        const int jj_P = pack_get_elem< proc_layout::template at< 1 >() >::apply(ii, jj, kk);                     \
        const int kk_P = pack_get_elem< proc_layout::template at< 2 >() >::apply(ii, jj, kk);                     \
        if ((ii != 0 || jj != 0 || kk != 0) && (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {         \
            _field0.pack(make_array(ii, jj, kk), _field0.ptr, it);                                                \
            operator()(hm, ii, jj, kk, it CALLNEXT_INST(n));                                                      \
        }                                                                                                         \
    }

            BOOST_PP_REPEAT(GCL_MAX_FIELDS, MACRO_IMPL, all)
#undef MACRO_IMPL
#undef CALLNEXT_INST
#undef _CALLNEXT_INST
#endif
        };

        template < int, int >
        struct unpack_dims {};

        template < int dummy >
        struct unpack_dims< 3, dummy > {

            template < typename T, typename iterator >
            void operator()(const T &, int, int, int, iterator &) const {}

#ifdef CXX11_ENABLED
            template < typename T, typename iterator, typename FIRST, typename... FIELDS >
            void operator()(
                const T &hm, int ii, int jj, int kk, iterator &it, FIRST const &first, const FIELDS &... _fields)
                const {
                typedef typename layout_transform< typename FIRST::inner_layoutmap, proc_layout_abs >::type proc_layout;
                const int ii_P = pack_get_elem< proc_layout::template at< 0 >() >::apply(ii, jj, kk);
                const int jj_P = pack_get_elem< proc_layout::template at< 1 >() >::apply(ii, jj, kk);
                const int kk_P = pack_get_elem< proc_layout::template at< 2 >() >::apply(ii, jj, kk);
                if ((ii != 0 || jj != 0 || kk != 0) && (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                    first.unpack(make_array(ii, jj, kk), first.ptr, it);
                    operator()(hm, ii, jj, kk, it, _fields...);
                }
            }
#else
//#define MBUILD(n) _field ## n
#define _CALLNEXT_INST(z, m, n) , _field##m
#define CALLNEXT_INST(m) BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(m), _CALLNEXT_INST, m)

#define MACRO_IMPL(z, n, _)                                                                                       \
    template < typename T, typename iterator, BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(n), typename FIELD) >        \
    void operator()(const T &hm,                                                                                  \
        int ii,                                                                                                   \
        int jj,                                                                                                   \
        int kk,                                                                                                   \
        iterator &it,                                                                                             \
        BOOST_PP_ENUM_BINARY_PARAMS_Z(z, BOOST_PP_INC(n), FIELD, const &_field)) const {                          \
        typedef typename layout_transform< typename FIELD0::inner_layoutmap, proc_layout_abs >::type proc_layout; \
        const int ii_P = pack_get_elem< proc_layout::template at< 0 >() >::apply(ii, jj, kk);                     \
        const int jj_P = pack_get_elem< proc_layout::template at< 1 >() >::apply(ii, jj, kk);                     \
        const int kk_P = pack_get_elem< proc_layout::template at< 2 >() >::apply(ii, jj, kk);                     \
        if ((ii != 0 || jj != 0 || kk != 0) && (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {         \
            _field0.unpack(make_array(ii, jj, kk), _field0.ptr, it);                                              \
            operator()(hm, ii, jj, kk, it CALLNEXT_INST(n));                                                      \
        }                                                                                                         \
    }

            BOOST_PP_REPEAT(GCL_MAX_FIELDS, MACRO_IMPL, all)

#undef MACRO_IMPL
#undef CALLNEXT_INST
#undef _CALLNEXT_INST
#endif
        };

        template < int, int >
        struct pack_vector_dims {};

        template < int dummy >
        struct pack_vector_dims< 3, dummy > {

            template < typename T, typename iterator, typename array_of_fotf >
            void operator()(const T &hm, int ii, int jj, int kk, iterator &it, array_of_fotf const &_fields) const {
                typedef typename layout_transform< typename array_of_fotf::value_type::inner_layoutmap,
                    proc_layout_abs >::type proc_layout;
                const int ii_P = pack_get_elem< proc_layout::template at< 0 >() >::apply(ii, jj, kk);
                const int jj_P = pack_get_elem< proc_layout::template at< 1 >() >::apply(ii, jj, kk);
                const int kk_P = pack_get_elem< proc_layout::template at< 2 >() >::apply(ii, jj, kk);
                if ((ii != 0 || jj != 0 || kk != 0) && (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                    for (unsigned int fi = 0; fi < _fields.size(); ++fi) {
                        _fields[fi].pack(make_array(ii, jj, kk), _fields[fi].ptr, it);
                    }
                }
            }
        };

        template < int, int >
        struct unpack_vector_dims {};

        template < int dummy >
        struct unpack_vector_dims< 3, dummy > {

            template < typename T, typename iterator, typename array_of_fotf >
            void operator()(const T &hm, int ii, int jj, int kk, iterator &it, array_of_fotf const &_fields) const {
                typedef typename layout_transform< typename array_of_fotf::value_type::inner_layoutmap,
                    proc_layout_abs >::type proc_layout;
                const int ii_P = pack_get_elem< proc_layout::template at< 0 >() >::apply(ii, jj, kk);
                const int jj_P = pack_get_elem< proc_layout::template at< 1 >() >::apply(ii, jj, kk);
                const int kk_P = pack_get_elem< proc_layout::template at< 2 >() >::apply(ii, jj, kk);
                if ((ii != 0 || jj != 0 || kk != 0) && (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                    for (unsigned int fi = 0; fi < _fields.size(); ++fi) {
                        _fields[fi].unpack(make_array(ii, jj, kk), _fields[fi].ptr, it);
                    }
                }
            }
        };
    };

#ifdef __CUDACC__
    template < typename HaloExch, typename proc_layout_abs >
    class hndlr_generic< 3, HaloExch, proc_layout_abs, gcl_gpu, version_manual > : public descriptor_base< HaloExch > {
        typedef gcl_gpu arch_type;

        static const int DIMS = 3;
        gridtools::array< char *, _impl::static_pow3< DIMS >::value > send_buffer; // One entry will not be used...
        gridtools::array< char *, _impl::static_pow3< DIMS >::value > recv_buffer;
        gridtools::array< int, _impl::static_pow3< DIMS >::value > send_buffer_size; // One entry will not be used...
        gridtools::array< int, _impl::static_pow3< DIMS >::value > recv_buffer_size;
        char **d_send_buffer;
        char **d_recv_buffer;

        int *prefix_send_size;
        int *prefix_recv_size;
        array< int, _impl::static_pow3< DIMS >::value > send_size;
        array< int, _impl::static_pow3< DIMS >::value > recv_size;

        int *d_send_size;
        int *d_recv_size;

        void *halo_d;   // pointer to halo descr on device
        void *halo_d_r; // pointer to halo descr on device

      public:
        typedef descriptor_base< HaloExch > base_type;
        typedef typename base_type::pattern_type pattern_type;

        /**
           Type of the computin grid associated to the pattern
         */
        typedef typename pattern_type::grid_type grid_type;

        /**
           Type of the translation used to map dimensions to buffer addresses
         */
        typedef translate_t< DIMS, typename default_layout_map< DIMS >::type > translate;

        hndlr_generic(grid_type const &g) : base_type(g) {}

        ~hndlr_generic() {
#ifdef _GCL_CHECK_DESTRUCTOR
            std::cout << "Destructor " << __FILE__ << ":" << __LINE__ << std::endl;
#endif

            for (int ii = -1; ii <= 1; ++ii)
                for (int jj = -1; jj <= 1; ++jj)
                    for (int kk = -1; kk <= 1; ++kk) {
                        _impl::gcl_alloc< char, arch_type >::free(send_buffer[translate()(ii, jj, kk)]);
                        _impl::gcl_alloc< char, arch_type >::free(recv_buffer[translate()(ii, jj, kk)]);
                    }
            delete[] prefix_send_size;
            delete[] prefix_recv_size;

            cudaFree(d_send_buffer);
            cudaFree(d_recv_buffer);
        }

        /**
           function to trigger data exchange

           Note: when the start_exchange() + wait() combination is used, the exchange() method should not be used, and
           vice versa.
        */
        void wait() { base_type::m_haloexch.wait(); }

        /**
           Setup function, in this version, takes tree parameters to
           compute internal buffers and sizes. It takes a field on the fly
           struct, which requires Datatype and layout map template
           arguments that are inferred, so the user is not aware of them.

           \tparam DataType This type is inferred by halo_example paramter
           \tparam data_layout This type is inferred by halo_example paramter

           \param[in] max_fields_n Maximum number of grids used in a computation
           \param[in] halo_example The (at least) maximal grid that is goinf to be used
           \param[in] typesize In case the DataType of the halo_example is not the same as the maximum data type used in
           the computation, this parameter can be given
         */
        template < typename DataType, typename f_data_layout, template < typename > class traits >
        void setup(int max_fields_n,
            field_on_the_fly< DataType, f_data_layout, traits > const &halo_example,
            int typesize = sizeof(DataType)) {
            typedef typename field_on_the_fly< DataType, f_data_layout, traits >::inner_layoutmap data_layout;
            prefix_send_size = new int[max_fields_n * 27];
            prefix_recv_size = new int[max_fields_n * 27];

            for (int ii = -1; ii <= 1; ++ii)
                for (int jj = -1; jj <= 1; ++jj)
                    for (int kk = -1; kk <= 1; ++kk)
                        if (ii != 0 || jj != 0 || kk != 0) {
                            typedef typename layout_transform< data_layout, proc_layout_abs >::type map_type;

                            const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                            const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                            const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);

                            if (base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1) {
                                send_size[translate()(ii, jj, kk)] =
                                    halo_example.send_buffer_size(make_array(ii, jj, kk));

                                send_buffer[translate()(ii, jj, kk)] = _impl::gcl_alloc< char, arch_type >::alloc(
                                    send_size[translate()(ii, jj, kk)] * max_fields_n * typesize);

                                base_type::m_haloexch.register_send_to_buffer(
                                    &(send_buffer[translate()(ii, jj, kk)][0]),
                                    send_size[translate()(ii, jj, kk)] * max_fields_n * typesize,
                                    ii_P,
                                    jj_P,
                                    kk_P);

                                recv_size[translate()(ii, jj, kk)] =
                                    halo_example.recv_buffer_size(make_array(ii, jj, kk));

                                recv_buffer[translate()(ii, jj, kk)] = _impl::gcl_alloc< char, arch_type >::alloc(
                                    recv_size[translate()(ii, jj, kk)] * max_fields_n * typesize);

                                base_type::m_haloexch.register_receive_from_buffer(
                                    &(recv_buffer[translate()(ii, jj, kk)][0]),
                                    recv_size[translate()(ii, jj, kk)] * max_fields_n * typesize,
                                    ii_P,
                                    jj_P,
                                    kk_P);

                                //(*filep) << "Size of buffer %d %d %d -> send %d -> recv %d" << ii << jj << kk <<
                                // send_size[translate()(ii,jj,kk)]*max_fields_n*typesize <<
                                // recv_size[translate()(ii,jj,kk)]*max_fields_n*typesize << std::endl;;

                            } else {
                                send_size[translate()(ii, jj, kk)] = 0;
                                send_buffer[translate()(ii, jj, kk)] = NULL;

                                base_type::m_haloexch.register_send_to_buffer(NULL, 0, ii_P, jj_P, kk_P);

                                recv_size[translate()(ii, jj, kk)] = 0;

                                recv_buffer[translate()(ii, jj, kk)] = NULL;

                                //(*filep) << "Size-of-buffer %d %d %d -> send %d -> recv %d" << ii << jj << kk <<
                                // send_size[translate()(ii,jj,kk)]*max_fields_n*typesize <<
                                // recv_size[translate()(ii,jj,kk)]*max_fields_n*typesize << std::endl;
                                base_type::m_haloexch.register_receive_from_buffer(NULL, 0, ii_P, jj_P, kk_P);
                            }
                        }

            cudaError_t err;
            err = cudaMalloc((&d_send_buffer), _impl::static_pow3< DIMS >::value * sizeof(DataType *));
            if (err != cudaSuccess) {
                printf("Error creating buffer table on device. Size: %d\n",
                    _impl::static_pow3< DIMS >::value * sizeof(DataType *));
            }

            err = cudaMemcpy(d_send_buffer,
                &(send_buffer[0]),
                _impl::static_pow3< DIMS >::value * sizeof(DataType *),
                cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                printf("Error transferring buffer table to device. Size: %d\n",
                    _impl::static_pow3< DIMS >::value * sizeof(DataType *));
            }

            err = cudaMalloc((&d_recv_buffer), _impl::static_pow3< DIMS >::value * sizeof(DataType *));
            if (err != cudaSuccess) {
                printf("Error creating buffer table (recv) on device. Size: %d\n",
                    _impl::static_pow3< DIMS >::value * sizeof(DataType *));
            }

            err = cudaMemcpy(d_recv_buffer,
                &(recv_buffer[0]),
                _impl::static_pow3< DIMS >::value * sizeof(DataType *),
                cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                printf("Error transferring buffer table (recv) to device. Size: %d\n",
                    _impl::static_pow3< DIMS >::value * sizeof(DataType *));
            }
        }

        /**
           Function to unpack received data

           \param[in] fields vector with data fields pointers to be packed from
        */
        template < typename T1, typename T2, template < typename > class T3 >
        void pack(std::vector< field_on_the_fly< T1, T2, T3 > > const &_fields) {

            typedef typename layout_transform< typename field_on_the_fly< T1, T2, T3 >::inner_layoutmap,
                proc_layout_abs >::type map_type;

            std::vector< field_on_the_fly< T1, T2, T3 > > fields = _fields;

            {
                int ii = 1;
                int jj = 0;
                int kk = 0;
                const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    for (int l = 0; l < fields.size(); ++l)
                        fields[l].halos[0].reset_minus();
                }
            }
            {
                int ii = -1;
                int jj = 0;
                int kk = 0;
                const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    for (int l = 0; l < fields.size(); ++l)
                        fields[l].halos[0].reset_plus();
                }
            }
            {
                int ii = 0;
                int jj = 1;
                int kk = 0;
                const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    for (int l = 0; l < fields.size(); ++l)
                        fields[l].halos[1].reset_minus();
                }
            }
            {
                int ii = 0;
                int jj = -1;
                int kk = 0;
                const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    for (int l = 0; l < fields.size(); ++l)
                        fields[l].halos[1].reset_plus();
                }
            }
            {
                int ii = 0;
                int jj = 0;
                int kk = 1;
                const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    for (int l = 0; l < fields.size(); ++l)
                        fields[l].halos[2].reset_minus();
                }
            }
            {
                int ii = 0;
                int jj = 0;
                int kk = -1;
                const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    for (int l = 0; l < fields.size(); ++l)
                        fields[l].halos[2].reset_plus();
                }
            }

            // for (int l=0; l<fields.size(); ++l)
            //   std::cout << "after trimming " << l << " " << fields[l] << std::endl;

            /* Computing the (prefix sums for) offsets to place fields in linear buffers
             */
            for (int ii = -1; ii <= 1; ++ii)
                for (int jj = -1; jj <= 1; ++jj)
                    for (int kk = -1; kk <= 1; ++kk) {
                        const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                        const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                        const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
                        if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                            if (ii != 0 || jj != 0 || kk != 0) {
                                prefix_send_size[0 + translate()(ii, jj, kk)] = 0;
                                // printf("prefix_send_size[l*27+translate()(ii,jj,kk)]=prefix_send_size[%d]=%d\n",0*27+translate()(ii,jj,kk),
                                // prefix_send_size[0*27+translate()(ii,jj,kk)]);
                                for (int l = 1; l < fields.size(); ++l) {
                                    prefix_send_size[l * 27 + translate()(ii, jj, kk)] =
                                        prefix_send_size[(l - 1) * 27 + translate()(ii, jj, kk)] +
                                        fields[l - 1].send_buffer_size(make_array(ii, jj, kk));
                                    // printf("prefix_send_size[l*27+translate()(ii,jj,kk)]=prefix_send_size[%d]=%d\n",l*27+translate()(ii,jj,kk),
                                    // prefix_send_size[l*27+translate()(ii,jj,kk)]);
                                }
                            }
                        }
                    }

            // typedef translate_t<3,default_layout_map<3>::type > translate;
            if (send_size[translate()(0, 0, -1)]) {
                m_packZL_generic(fields,
                    reinterpret_cast< typename field_on_the_fly< T1, T2, T3 >::value_type ** >(d_send_buffer),
                    &(prefix_send_size[0]));
            }
            if (send_size[translate()(0, 0, 1)]) {
                m_packZU_generic(fields,
                    reinterpret_cast< typename field_on_the_fly< T1, T2, T3 >::value_type ** >(d_send_buffer),
                    &(prefix_send_size[0]));
            }
            if (send_size[translate()(0, -1, 0)]) {
                m_packYL_generic(fields,
                    reinterpret_cast< typename field_on_the_fly< T1, T2, T3 >::value_type ** >(d_send_buffer),
                    &(prefix_send_size[0]));
            }
            if (send_size[translate()(0, 1, 0)]) {
                m_packYU_generic(fields,
                    reinterpret_cast< typename field_on_the_fly< T1, T2, T3 >::value_type ** >(d_send_buffer),
                    &(prefix_send_size[0]));
            }
            if (send_size[translate()(-1, 0, 0)]) {
                m_packXL_generic(fields,
                    reinterpret_cast< typename field_on_the_fly< T1, T2, T3 >::value_type ** >(d_send_buffer),
                    &(prefix_send_size[0]));
            }
            if (send_size[translate()(1, 0, 0)]) {
                m_packXU_generic(fields,
                    reinterpret_cast< typename field_on_the_fly< T1, T2, T3 >::value_type ** >(d_send_buffer),
                    &(prefix_send_size[0]));
            }

#ifdef GCL_MULTI_STREAMS
            cudaStreamSynchronize(ZL_stream);
            cudaStreamSynchronize(ZU_stream);
            cudaStreamSynchronize(YL_stream);
            cudaStreamSynchronize(YU_stream);
            cudaStreamSynchronize(XL_stream);
            cudaStreamSynchronize(XU_stream);
#else
            cudaDeviceSynchronize();
#endif
        }

        /**
           Function to unpack received data

           \param[in] fields vector with data fields pointers to be unpacked into
        */
        template < typename T1, typename T2, template < typename > class T3 >
        void unpack(std::vector< field_on_the_fly< T1, T2, T3 > > const &_fields) {
            typedef typename layout_transform< typename field_on_the_fly< T1, T2, T3 >::inner_layoutmap,
                proc_layout_abs >::type map_type;

            std::vector< field_on_the_fly< T1, T2, T3 > > fields = _fields;

            {
                int ii = 1;
                int jj = 0;
                int kk = 0;
                const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    for (int l = 0; l < fields.size(); ++l)
                        fields[l].halos[0].reset_plus();
                }
            }
            {
                int ii = -1;
                int jj = 0;
                int kk = 0;
                const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    for (int l = 0; l < fields.size(); ++l)
                        fields[l].halos[0].reset_minus();
                }
            }
            {
                int ii = 0;
                int jj = 1;
                int kk = 0;
                const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    for (int l = 0; l < fields.size(); ++l)
                        fields[l].halos[1].reset_plus();
                }
            }
            {
                int ii = 0;
                int jj = -1;
                int kk = 0;
                const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    for (int l = 0; l < fields.size(); ++l)
                        fields[l].halos[1].reset_minus();
                }
            }
            {
                int ii = 0;
                int jj = 0;
                int kk = 1;
                const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    for (int l = 0; l < fields.size(); ++l)
                        fields[l].halos[2].reset_plus();
                }
            }
            {
                int ii = 0;
                int jj = 0;
                int kk = -1;
                const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    for (int l = 0; l < fields.size(); ++l)
                        fields[l].halos[2].reset_minus();
                }
            }

            for (int ii = -1; ii <= 1; ++ii)
                for (int jj = -1; jj <= 1; ++jj)
                    for (int kk = -1; kk <= 1; ++kk) {
                        const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                        const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                        const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
                        if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                            if (ii != 0 || jj != 0 || kk != 0) {
                                prefix_recv_size[0 + translate()(ii, jj, kk)] = 0;
                                for (int l = 1; l < fields.size(); ++l) {
                                    prefix_recv_size[l * 27 + translate()(ii, jj, kk)] =
                                        prefix_recv_size[(l - 1) * 27 + translate()(ii, jj, kk)] +
                                        fields[l - 1].recv_buffer_size(make_array(ii, jj, kk));
                                }
                            }
                        }
                    }

            // typedef translate_t<3,default_layout_map<3>::type > translate;
            if (recv_size[translate()(0, 0, -1)]) {
                m_unpackZL_generic(fields,
                    reinterpret_cast< typename field_on_the_fly< T1, T2, T3 >::value_type ** >(d_recv_buffer),
                    &(prefix_recv_size[0]));
            }
            if (recv_size[translate()(0, 0, 1)]) {
                m_unpackZU_generic(fields,
                    reinterpret_cast< typename field_on_the_fly< T1, T2, T3 >::value_type ** >(d_recv_buffer),
                    &(prefix_recv_size[0]));
            }
            if (recv_size[translate()(0, -1, 0)]) {
                m_unpackYL_generic(fields,
                    reinterpret_cast< typename field_on_the_fly< T1, T2, T3 >::value_type ** >(d_recv_buffer),
                    &(prefix_recv_size[0]));
            }
            if (recv_size[translate()(0, 1, 0)]) {
                m_unpackYU_generic(fields,
                    reinterpret_cast< typename field_on_the_fly< T1, T2, T3 >::value_type ** >(d_recv_buffer),
                    &(prefix_recv_size[0]));
            }
            if (recv_size[translate()(-1, 0, 0)]) {
                m_unpackXL_generic(fields,
                    reinterpret_cast< typename field_on_the_fly< T1, T2, T3 >::value_type ** >(d_recv_buffer),
                    &(prefix_recv_size[0]));
            }
            if (recv_size[translate()(1, 0, 0)]) {
                m_unpackXU_generic(fields,
                    reinterpret_cast< typename field_on_the_fly< T1, T2, T3 >::value_type ** >(d_recv_buffer),
                    &(prefix_recv_size[0]));
            }
        }

#include <communication/high-level/non_vect_interface.hpp>
    };
#endif // cudacc
}

#endif
