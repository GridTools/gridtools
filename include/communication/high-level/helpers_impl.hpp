/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#ifndef _HELPERS_IMPL_H_
#define _HELPERS_IMPL_H_
#include "../../common/generic_metafunctions/pack_get_elem.hpp"

namespace gridtools {
    namespace _impl {

        template < typename T, typename arch /*=gcl_cpu*/ >
        struct gcl_alloc;

        template < typename T >
        struct gcl_alloc< T, gcl_cpu > {

            static T *alloc(size_t sz) {
                if (sz)
                    return new T[sz];
                else
                    return NULL;
            }

            static void free(T *t) {
                if (!t)
                    delete[] t;
                t = NULL;
            }
        };

#ifdef _GCL_GPU_
        template < typename T >
        struct gcl_alloc< T, gcl_gpu > {

            static T *alloc(size_t sz) {
                if (sz) {
                    T *ptr;
                    cudaError_t status = cudaMalloc(&ptr, sz * sizeof(T));
                    if (!checkCudaStatus(status)) {
                        printf("Allocation did not succed\n");
                        exit(1);
                    }
                    return ptr;
                } else {
                    return NULL;
                }
            }

            static void free(T *t) {
                if (!t)
                    cudaError_t status = cudaFree(t);
                t = NULL;
            }
        };
#endif

        template < typename T >
        struct allocation_service;

        template < typename Datatype, typename T2 >
        struct allocation_service< hndlr_descriptor_ut< Datatype, 3, T2 > > {
            void operator()(hndlr_descriptor_ut< Datatype, 3, T2 > *hm) const {
                typedef typename hndlr_descriptor_ut< Datatype, 3, T2 >::pattern_type::translate_type translate;
                for (int ii = -1; ii <= 1; ++ii)
                    for (int jj = -1; jj <= 1; ++jj)
                        for (int kk = -1; kk <= 1; ++kk)
                            if (ii != 0 || jj != 0 || kk != 0) {
                                // std::cout << hm->total_pack_size(make_array(ii,jj,kk)) << " " <<
                                // hm->total_unpack_size(make_array(ii,jj,kk)) << "\n";
                                hm->send_buffer[translate()(ii, jj, kk)] = _impl::gcl_alloc< Datatype, gcl_cpu >::alloc(
                                    hm->total_pack_size(make_array(ii, jj, kk)));
                                hm->recv_buffer[translate()(ii, jj, kk)] = _impl::gcl_alloc< Datatype, gcl_cpu >::alloc(
                                    hm->total_unpack_size(make_array(ii, jj, kk)));

                                hm->m_haloexch.register_send_to_buffer(&(hm->send_buffer[translate()(ii, jj, kk)][0]),
                                    hm->total_pack_size(make_array(ii, jj, kk)) * sizeof(Datatype),
                                    ii,
                                    jj,
                                    kk);

                                hm->m_haloexch.register_receive_from_buffer(
                                    &(hm->recv_buffer[translate()(ii, jj, kk)][0]),
                                    hm->total_unpack_size(make_array(ii, jj, kk)) * sizeof(Datatype),
                                    ii,
                                    jj,
                                    kk);
                            }
            }
        };

        template < typename Datatype,
            typename T2,
            typename procmap,
            typename arch,
            int V,
            template < int Ndim > class GridType >
        struct allocation_service< hndlr_dynamic_ut< Datatype, GridType< 3 >, T2, procmap, arch, V > > {
            void operator()(hndlr_dynamic_ut< Datatype, GridType< 3 >, T2, procmap, arch, V > *hm, int mf) const {
                typedef translate_t< 3, default_layout_map< 3 >::type > translate;
                typedef translate_t< 3, procmap > translate_P;

                for (int ii = -1; ii <= 1; ++ii)
                    for (int jj = -1; jj <= 1; ++jj)
                        for (int kk = -1; kk <= 1; ++kk)
                            if (ii != 0 || jj != 0 || kk != 0) {
                                // std::cout << hm->total_pack_size(make_array(ii,jj,kk)) << " " <<
                                // hm->total_unpack_size(make_array(ii,jj,kk)) << "\n";
                                hm->send_buffer[translate()(ii, jj, kk)] = _impl::gcl_alloc< Datatype, arch >::alloc(
                                    hm->halo.send_buffer_size(make_array(ii, jj, kk)) * mf);
                                hm->recv_buffer[translate()(ii, jj, kk)] = _impl::gcl_alloc< Datatype, arch >::alloc(
                                    hm->halo.recv_buffer_size(make_array(ii, jj, kk)) * mf);

                                typedef typename translate_P::map_type map_type;
                                const int ii_P = pack_get_elem<map_type::template at<0>()>::apply(ii, jj, kk);
                                const int jj_P = pack_get_elem<map_type::template at<1>()>::apply(ii, jj, kk);
                                const int kk_P = pack_get_elem<map_type::template at<2>()>::apply(ii, jj, kk);

                                hm->m_haloexch.register_send_to_buffer(&(hm->send_buffer[translate()(ii, jj, kk)][0]),
                                    hm->halo.send_buffer_size(make_array(ii, jj, kk)) * sizeof(Datatype) * mf,
                                    ii_P,
                                    jj_P,
                                    kk_P);

                                hm->m_haloexch.register_receive_from_buffer(
                                    &(hm->recv_buffer[translate()(ii, jj, kk)][0]),
                                    hm->halo.recv_buffer_size(make_array(ii, jj, kk)) * sizeof(Datatype) * mf,
                                    ii_P,
                                    jj_P,
                                    kk_P);
                            }
            }
        };

        template < typename T >
        struct pack_service;

        template < typename Datatype, typename T2 >
        struct pack_service< hndlr_descriptor_ut< Datatype, 3, T2 > > {
            void operator()(hndlr_descriptor_ut< Datatype, 3, T2 > const *hm) const {
                typedef typename hndlr_descriptor_ut< Datatype, 3, T2 >::pattern_type::translate_type translate;
                for (int ii = -1; ii <= 1; ++ii)
                    for (int jj = -1; jj <= 1; ++jj)
                        for (int kk = -1; kk <= 1; ++kk)
                            if ((ii != 0 || jj != 0 || kk != 0) && (hm->pattern().proc_grid().proc(ii, jj, kk) != -1)) {
                                Datatype *it = &(hm->send_buffer[translate()(ii, jj, kk)][0]);
                                for (int df = 0; df < hm->size(); ++df)
                                    hm->data_field(df).pack(make_array(ii, jj, kk), it);
                            }
            }
        };

        template < typename T >
        struct unpack_service;

        template < typename Datatype, typename T2 >
        struct unpack_service< hndlr_descriptor_ut< Datatype, 3, T2 > > {
            void operator()(hndlr_descriptor_ut< Datatype, 3, T2 > const *hm) const {
                typedef typename hndlr_descriptor_ut< Datatype, 3, T2 >::pattern_type::translate_type translate;
                for (int ii = -1; ii <= 1; ++ii)
                    for (int jj = -1; jj <= 1; ++jj)
                        for (int kk = -1; kk <= 1; ++kk)
                            if ((ii != 0 || jj != 0 || kk != 0) && (hm->pattern().proc_grid().proc(ii, jj, kk) != -1)) {
                                Datatype *it = &(hm->recv_buffer[translate()(ii, jj, kk)][0]);
                                for (int df = 0; df < hm->size(); ++df)
                                    hm->data_field(df).unpack(make_array(ii, jj, kk), it);
                            }
            }
        };

    } // namespace _impl
} // namespace gridtools
#endif
