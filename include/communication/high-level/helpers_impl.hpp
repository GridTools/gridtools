#ifndef _HELPERS_IMPL_H_
#define _HELPERS_IMPL_H_

namespace gridtools {
    namespace _impl {

        template <typename T, typename arch/*=gcl_cpu*/>
        struct gcl_alloc;

        template <typename T>
        struct gcl_alloc<T,gcl_cpu> {

            static
            T* alloc(size_t sz) {
                if (sz)
                    return new T[sz];
                else
                    return NULL;
            }

            static
            void free(T* t) {
                if (!t)
                    delete [] t;
                t = NULL;
            }
        };

#ifdef _GCL_GPU_
        template <typename T>
        struct gcl_alloc<T,gcl_gpu> {

            static
            T* alloc(size_t sz) {
                if (sz) {
                    T* ptr;
                    cudaError_t status = cudaMalloc( &ptr, sz * sizeof( T ) );
                    if( !checkCudaStatus( status ) ) {
                        printf("Allocation did not succed\n");
                        exit(1);
                    }
                    return ptr;
                } else {
                    return NULL;
                }
            }

            static
            void free(T* t) {
                if (!t)
                    cudaError_t status = cudaFree( t );
                t = NULL;
            }
        };
#endif

        template <typename T>
        struct allocation_service;


        template <typename Datatype, typename T2>
        struct allocation_service<hndlr_descriptor_ut<Datatype, 3, T2> > {
            void operator()(hndlr_descriptor_ut<Datatype, 3, T2> *hm) const {
                typedef typename hndlr_descriptor_ut<Datatype, 3, T2>::pattern_type::translate_type translate;
                for (int ii=-1; ii<=1; ++ii)
                    for (int jj=-1; jj<=1; ++jj)
                        for (int kk=-1; kk<=1; ++kk)
                            if (ii!=0 || jj!=0 || kk!=0) {
                                //std::cout << hm->total_pack_size(make_array(ii,jj,kk)) << " " << hm->total_unpack_size(make_array(ii,jj,kk)) << "\n";
                                hm->send_buffer[translate()(ii,jj,kk)] =
                                    _impl::gcl_alloc<Datatype, gcl_cpu>::alloc(hm->total_pack_size(make_array(ii,jj,kk)));
                                hm->recv_buffer[translate()(ii,jj,kk)] =
                                    _impl::gcl_alloc<Datatype, gcl_cpu>::alloc(hm->total_unpack_size(make_array(ii,jj,kk)));

                                hm->m_haloexch.register_send_to_buffer
                                    (&(hm->send_buffer[translate()(ii,jj,kk)][0]),
                                     hm->total_pack_size(make_array(ii,jj,kk))*sizeof(Datatype), ii, jj, kk);

                                hm->m_haloexch.register_receive_from_buffer
                                    (&(hm->recv_buffer[translate()(ii,jj,kk)][0]),
                                     hm->total_unpack_size(make_array(ii,jj,kk))*sizeof(Datatype),ii,jj,kk);
                            }
            }
        };

        template <typename Datatype, typename T2, typename procmap, typename arch, int V,
                  template <int Ndim>
                  class GridType>
        struct allocation_service<hndlr_dynamic_ut<Datatype, GridType<3>, T2, procmap, arch, V> > {
            void operator()(hndlr_dynamic_ut<Datatype, GridType<3>, T2, procmap, arch, V> *hm, int mf) const {
                typedef translate_t<3,default_layout_map<3>::type > translate;
                typedef translate_t<3,procmap> translate_P;

                for (int ii=-1; ii<=1; ++ii)
                    for (int jj=-1; jj<=1; ++jj)
                        for (int kk=-1; kk<=1; ++kk)
                            if (ii!=0 || jj!=0 || kk!=0) {
                                //std::cout << hm->total_pack_size(make_array(ii,jj,kk)) << " " << hm->total_unpack_size(make_array(ii,jj,kk)) << "\n";
                                hm->send_buffer[translate()(ii,jj,kk)] =
                                    _impl::gcl_alloc<Datatype,arch>::alloc(hm->halo.send_buffer_size(make_array(ii,jj,kk))*mf);
                                hm->recv_buffer[translate()(ii,jj,kk)] =
                                    _impl::gcl_alloc<Datatype,arch>::alloc(hm->halo.recv_buffer_size(make_array(ii,jj,kk))*mf);

                                typedef typename translate_P::map_type map_type;
                                const int ii_P = map_type().template select<0>(ii,jj,kk);
                                const int jj_P = map_type().template select<1>(ii,jj,kk);
                                const int kk_P = map_type().template select<2>(ii,jj,kk);

                                hm->m_haloexch.register_send_to_buffer
                                    (&(hm->send_buffer[translate()(ii,jj,kk)][0]),
                                     hm->halo.send_buffer_size(make_array(ii,jj,kk))*sizeof(Datatype)*mf,
                                     ii_P,jj_P,kk_P);

                                hm->m_haloexch.register_receive_from_buffer
                                    (&(hm->recv_buffer[translate()(ii,jj,kk)][0]),
                                     hm->halo.recv_buffer_size(make_array(ii,jj,kk))*sizeof(Datatype)*mf,
                                     ii_P,jj_P,kk_P);
                            }
            }
        };

        template <typename T>
        struct pack_service;

        template <typename Datatype, typename T2>
        struct pack_service<hndlr_descriptor_ut<Datatype, 3, T2> > {
            void operator()(hndlr_descriptor_ut<Datatype, 3, T2> const *hm) const {
                typedef typename hndlr_descriptor_ut<Datatype, 3, T2>::pattern_type::translate_type translate;
                for (int ii=-1; ii<=1; ++ii)
                    for (int jj=-1; jj<=1; ++jj)
                        for (int kk=-1; kk<=1; ++kk)
                            if ((ii!=0 || jj!=0 || kk!=0) && (hm->pattern().proc_grid().proc(ii,jj,kk) != -1)) {
                                Datatype *it = &(hm->send_buffer[translate()(ii,jj,kk)][0]);
                                for (int df = 0; df < hm->size(); ++df)
                                    hm->data_field(df).pack(make_array(ii,jj,kk), it);
                            }
            }
        };

        template <typename T>
        struct unpack_service;

        template <typename Datatype, typename T2>
        struct unpack_service<hndlr_descriptor_ut<Datatype, 3, T2> > {
            void operator()(hndlr_descriptor_ut<Datatype, 3, T2> const *hm) const {
                typedef typename hndlr_descriptor_ut<Datatype, 3, T2>::pattern_type::translate_type translate;
                for (int ii=-1; ii<=1; ++ii)
                    for (int jj=-1; jj<=1; ++jj)
                        for (int kk=-1; kk<=1; ++kk)
                            if ((ii!=0 || jj!=0 || kk!=0) && (hm->pattern().proc_grid().proc(ii,jj,kk) != -1)) {
                                Datatype *it = &(hm->recv_buffer[translate()(ii,jj,kk)][0]);
                                for (int df = 0; df < hm->size(); ++df)
                                    hm->data_field(df).unpack(make_array(ii,jj,kk), it);
                            }
            }
        };


    } // namespace _impl
} // namespace gridtools
#endif
