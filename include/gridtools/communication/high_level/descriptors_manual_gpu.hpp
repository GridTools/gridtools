/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../../meta/utility.hpp"

#ifdef __CUDACC__
#include "m_packXL.hpp"
#include "m_packXU.hpp"
#include "m_packYL.hpp"
#include "m_packYU.hpp"
#include "m_packZL.hpp"
#include "m_packZU.hpp"

#include "m_unpackXL.hpp"
#include "m_unpackXU.hpp"
#include "m_unpackYL.hpp"
#include "m_unpackYU.hpp"
#include "m_unpackZL.hpp"
#include "m_unpackZU.hpp"
#endif

namespace gridtools {
    /** \class empty_field_no_dt_gpu
        Class containint the information about a data field (grid).
        It doe not contains any reference to actual data of the field,
        it only describes the fields though the halo descriptions.
        The number of dimensions as a template argument and the size of the
        first dimension, the size of the non-halo data field,
        the halo width before and after the actual data, then the same for the
        second dimension, the third, etc. This information is encoded in
        halo_descriptor. A dimension of the field is described as:
        \code
        |-----|------|---------------|---------|----|
        | pad0|minus |    length     | plus    |pad1|
                      ^begin        ^end
        |               total_length                |
        \endcode

        \tparam DIMS the number of dimensions of the data field
    */
    template <int DIMS>
    class empty_field_no_dt_gpu : public empty_field_base<int> {

        typedef empty_field_base<int> base_type;

      public:
        /**
            Constructor that receive the pointer to the data. This is explicit and
            must then be called.
        */
        explicit empty_field_no_dt_gpu() {}

        void setup() const {}

        const halo_descriptor *raw_array() const { return &(base_type::halos[0]); }

      protected:
        template <typename DataType, typename GridType, typename HaloExch, typename proc_layout, typename arch>
        friend class hndlr_dynamic_ut;

        template <int I>
        friend std::ostream &operator<<(std::ostream &s, empty_field_no_dt_gpu<I> const &ef);

        halo_descriptor *dangerous_raw_array() { return &(base_type::halos[0]); }
    };

    template <int I>
    std::ostream &operator<<(std::ostream &s, empty_field_no_dt_gpu<I> const &ef) {
        s << "empty_field_no_dt_gpu ";
        for (int i = 0; i < I; ++i)
            s << ef.raw_array()[i] << ", ";
        return s;
    }

#ifdef __CUDACC__
    /** specialization for GPU and manual packing */
    template <typename DataType, typename HaloExch, typename proc_layout, template <int Ndim> class GridType>
    class hndlr_dynamic_ut<DataType, GridType<3>, HaloExch, proc_layout, gcl_gpu> : public descriptor_base<HaloExch> {

        static const int DIMS = 3;

        typedef hndlr_dynamic_ut<DataType, GridType<3>, HaloExch, proc_layout, gcl_gpu> this_type;

      public:
        empty_field_no_dt_gpu<DIMS> halo;

      private:
        typedef gcl_gpu arch_type;
        DataType **d_send_buffer;
        DataType **d_recv_buffer;

        halo_descriptor dangeroushalo[3];
        halo_descriptor dangeroushalo_r[3];
        array<DataType *, _impl::static_pow3<DIMS>::value> send_buffer;
        array<DataType *, _impl::static_pow3<DIMS>::value> recv_buffer;
        array<int, _impl::static_pow3<DIMS>::value> send_size;
        array<int, _impl::static_pow3<DIMS>::value> recv_size;
        int *d_send_size;
        int *d_recv_size;

        halo_descriptor *halo_d;   // pointer to halo descr on device
        halo_descriptor *halo_d_r; // pointer to halo descr on device
      public:
        typedef descriptor_base<HaloExch> base_type;
        typedef base_type pattern_type;

        /**
           Type of the computin grid associated to the pattern
         */
        typedef typename pattern_type::grid_type grid_type;

        /**
           Type of the translation used to map dimensions to buffer addresses
         */
        typedef translate_t<DIMS, typename default_layout_map<DIMS>::type> translate;

      private:
        hndlr_dynamic_ut(hndlr_dynamic_ut const &) = delete;
        hndlr_dynamic_ut(hndlr_dynamic_ut &&) = delete;

      public:
#ifdef GCL_TRACE
        void set_pattern_tag(int tag) { base_type::m_haloexch.set_pattern_tag(tag); };
#endif
        /**
           Constructor

           \param[in] c The object of the class used to specify periodicity in each dimension
           \param[in] comm MPI communicator (typically MPI_Comm_world)
        */
        explicit hndlr_dynamic_ut(typename grid_type::period_type const &c, MPI_Comm const &comm)
            : base_type(c, comm), send_buffer{nullptr}, recv_buffer{nullptr}, send_size{0}, recv_size{0} {}

        /**
           Constructor

           \param[in] g A processor grid that will execute the pattern
         */
        explicit hndlr_dynamic_ut(grid_type const &g)
            : base_type(g), send_buffer{nullptr}, recv_buffer{nullptr}, send_size{0}, recv_size{0} {}

        ~hndlr_dynamic_ut() {
#ifdef GCL_CHECK_DESTRUCTOR
            std::cout << "Destructor " << __FILE__ << ":" << __LINE__ << std::endl;
#endif

            for (int i = -1; i <= 1; ++i)
                for (int j = -1; j <= 1; ++j)
                    for (int k = -1; k <= 1; ++k) {
                        _impl::gcl_alloc<DataType, arch_type>::free(send_buffer[translate()(i, j, k)]);
                        _impl::gcl_alloc<DataType, arch_type>::free(recv_buffer[translate()(i, j, k)]);
                    }
            GT_CUDA_CHECK(cudaFree(d_send_buffer));
            GT_CUDA_CHECK(cudaFree(d_recv_buffer));
            GT_CUDA_CHECK(cudaFree(d_send_size));
            GT_CUDA_CHECK(cudaFree(d_recv_size));
            GT_CUDA_CHECK(cudaFree(halo_d));
            GT_CUDA_CHECK(cudaFree(halo_d_r));
        }

        /**
           Function to setup internal data structures for data exchange and preparing eventual underlying layers

           \param max_fields_n Maximum number of data fields that will be passed to the communication functions
        */
        void setup(const int max_fields_n) {

            typedef translate_t<3, default_layout_map<3>::type> translate;
            typedef translate_t<3, proc_layout> translate_P;

            dangeroushalo[0] = halo.dangerous_raw_array()[0];
            dangeroushalo[1] = halo.dangerous_raw_array()[1];
            dangeroushalo[2] = halo.dangerous_raw_array()[2];
            dangeroushalo_r[0] = halo.dangerous_raw_array()[0];
            dangeroushalo_r[1] = halo.dangerous_raw_array()[1];
            dangeroushalo_r[2] = halo.dangerous_raw_array()[2];

            // printf("%d danhalo %d %d %d %d %d, %d %d %d %d %d, %d %d %d %d %d\n", PID,
            //        dangeroushalo[0].minus(),
            //        dangeroushalo[0].plus(),
            //        dangeroushalo[0].begin(),
            //        dangeroushalo[0].end(),
            //        dangeroushalo[0].total_length(),
            //        dangeroushalo[1].minus(),
            //        dangeroushalo[1].plus(),
            //        dangeroushalo[1].begin(),
            //        dangeroushalo[1].end(),
            //        dangeroushalo[1].total_length(),
            //        dangeroushalo[2].minus(),
            //        dangeroushalo[2].plus(),
            //        dangeroushalo[2].begin(),
            //        dangeroushalo[2].end(),
            //        dangeroushalo[2].total_length());
            // printf("%d danhalo_r %d %d %d %d %d, %d %d %d %d %d, %d %d %d %d %d\n", PID,
            //        dangeroushalo_r[0].minus(),
            //        dangeroushalo_r[0].plus(),
            //        dangeroushalo_r[0].begin(),
            //        dangeroushalo_r[0].end(),
            //        dangeroushalo_r[0].total_length(),
            //        dangeroushalo_r[1].minus(),
            //        dangeroushalo_r[1].plus(),
            //        dangeroushalo_r[1].begin(),
            //        dangeroushalo_r[1].end(),
            //        dangeroushalo_r[1].total_length(),
            //        dangeroushalo_r[2].minus(),
            //        dangeroushalo_r[2].plus(),
            //        dangeroushalo_r[2].begin(),
            //        dangeroushalo_r[2].end(),
            //        dangeroushalo_r[2].total_length());
            {
                typedef proc_layout map_type;
                int ii = 1;
                int jj = 0;
                int kk = 0;
                const int ii_P = pack_get_elem<map_type::template at<0>()>::apply(ii, jj, kk);
                const int jj_P = pack_get_elem<map_type::template at<1>()>::apply(ii, jj, kk);
                const int kk_P = pack_get_elem<map_type::template at<2>()>::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    dangeroushalo[0].reset_minus();
                    dangeroushalo_r[0].reset_plus();
                }
            }
            {
                typedef proc_layout map_type;
                int ii = -1;
                int jj = 0;
                int kk = 0;
                const int ii_P = pack_get_elem<map_type::template at<0>()>::apply(ii, jj, kk);
                const int jj_P = pack_get_elem<map_type::template at<1>()>::apply(ii, jj, kk);
                const int kk_P = pack_get_elem<map_type::template at<2>()>::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    dangeroushalo[0].reset_plus();
                    dangeroushalo_r[0].reset_minus();
                }
            }
            {
                typedef proc_layout map_type;
                int ii = 0;
                int jj = 1;
                int kk = 0;
                const int ii_P = pack_get_elem<map_type::template at<0>()>::apply(ii, jj, kk);
                const int jj_P = pack_get_elem<map_type::template at<1>()>::apply(ii, jj, kk);
                const int kk_P = pack_get_elem<map_type::template at<2>()>::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    dangeroushalo[1].reset_minus();
                    dangeroushalo_r[1].reset_plus();
                }
            }
            {
                typedef proc_layout map_type;
                int ii = 0;
                int jj = -1;
                int kk = 0;
                const int ii_P = pack_get_elem<map_type::template at<0>()>::apply(ii, jj, kk);
                const int jj_P = pack_get_elem<map_type::template at<1>()>::apply(ii, jj, kk);
                const int kk_P = pack_get_elem<map_type::template at<2>()>::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    dangeroushalo[1].reset_plus();
                    dangeroushalo_r[1].reset_minus();
                }
            }
            {
                typedef proc_layout map_type;
                int ii = 0;
                int jj = 0;
                int kk = 1;
                const int ii_P = pack_get_elem<map_type::template at<0>()>::apply(ii, jj, kk);
                const int jj_P = pack_get_elem<map_type::template at<1>()>::apply(ii, jj, kk);
                const int kk_P = pack_get_elem<map_type::template at<2>()>::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    dangeroushalo[2].reset_minus();
                    dangeroushalo_r[2].reset_plus();
                }
            }
            {
                typedef proc_layout map_type;
                int ii = 0;
                int jj = 0;
                int kk = -1;
                const int ii_P = pack_get_elem<map_type::template at<0>()>::apply(ii, jj, kk);
                const int jj_P = pack_get_elem<map_type::template at<1>()>::apply(ii, jj, kk);
                const int kk_P = pack_get_elem<map_type::template at<2>()>::apply(ii, jj, kk);
                if ((base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) == -1)) {
                    dangeroushalo[2].reset_plus();
                    dangeroushalo_r[2].reset_minus();
                }
            }

            // printf("%d danhalo %d %d %d %d %d, %d %d %d %d %d, %d %d %d %d %d\n", PID,
            //        dangeroushalo[0].minus(),
            //        dangeroushalo[0].plus(),
            //        dangeroushalo[0].begin(),
            //        dangeroushalo[0].end(),
            //        dangeroushalo[0].total_length(),
            //        dangeroushalo[1].minus(),
            //        dangeroushalo[1].plus(),
            //        dangeroushalo[1].begin(),
            //        dangeroushalo[1].end(),
            //        dangeroushalo[1].total_length(),
            //        dangeroushalo[2].minus(),
            //        dangeroushalo[2].plus(),
            //        dangeroushalo[2].begin(),
            //        dangeroushalo[2].end(),
            //        dangeroushalo[2].total_length());

            // printf("%d danhalo_r %d %d %d %d %d, %d %d %d %d %d, %d %d %d %d %d\n", PID,
            //        dangeroushalo_r[0].minus(),
            //        dangeroushalo_r[0].plus(),
            //        dangeroushalo_r[0].begin(),
            //        dangeroushalo_r[0].end(),
            //        dangeroushalo_r[0].total_length(),
            //        dangeroushalo_r[1].minus(),
            //        dangeroushalo_r[1].plus(),
            //        dangeroushalo_r[1].begin(),
            //        dangeroushalo_r[1].end(),
            //        dangeroushalo_r[1].total_length(),
            //        dangeroushalo_r[2].minus(),
            //        dangeroushalo_r[2].plus(),
            //        dangeroushalo_r[2].begin(),
            //        dangeroushalo_r[2].end(),
            //        dangeroushalo_r[2].total_length());

            //       printf("halo 1 is: %d, %d, %d, %d, %d \n", (halo.halos[0]).minus(), (halo.halos[0]).plus(),
            //       (halo.halos[0]).begin(), (halo.halos[0]).end(), (halo.halos[0]).total_length());
            //       printf("halo 2 is: %d, %d, %d, %d, %d \n", (halo.halos[1]).minus(), (halo.halos[1]).plus(),
            //       (halo.halos[1]).begin(), (halo.halos[1]).end(), (halo.halos[1]).total_length());
            //       printf("halo 3 is: %d, %d, %d, %d, %d \n", (halo.halos[2]).minus(), (halo.halos[2]).plus(),
            //       (halo.halos[2]).begin(), (halo.halos[2]).end(), (halo.halos[2]).total_length());

            for (int ii = -1; ii <= 1; ++ii)
                for (int jj = -1; jj <= 1; ++jj)
                    for (int kk = -1; kk <= 1; ++kk)
                        if (ii != 0 || jj != 0 || kk != 0) {
                            typedef typename translate_P::map_type map_type;
                            const int ii_P = pack_get_elem<map_type::template at<0>()>::apply(ii, jj, kk);
                            const int jj_P = pack_get_elem<map_type::template at<1>()>::apply(ii, jj, kk);
                            const int kk_P = pack_get_elem<map_type::template at<2>()>::apply(ii, jj, kk);

                            if (base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1) {
                                send_size[translate()(ii, jj, kk)] = halo.send_buffer_size(make_array(ii, jj, kk));

                                send_buffer[translate()(ii, jj, kk)] = _impl::gcl_alloc<DataType, arch_type>::alloc(
                                    send_size[translate()(ii, jj, kk)] * max_fields_n);

                                base_type::m_haloexch.register_send_to_buffer(
                                    &(send_buffer[translate()(ii, jj, kk)][0]),
                                    send_size[translate()(ii, jj, kk)] * max_fields_n * sizeof(DataType),
                                    ii_P,
                                    jj_P,
                                    kk_P);

                                recv_size[translate()(ii, jj, kk)] = halo.recv_buffer_size(make_array(ii, jj, kk));

                                recv_buffer[translate()(ii, jj, kk)] = _impl::gcl_alloc<DataType, arch_type>::alloc(
                                    recv_size[translate()(ii, jj, kk)] * max_fields_n);

                                base_type::m_haloexch.register_receive_from_buffer(
                                    &(recv_buffer[translate()(ii, jj, kk)][0]),
                                    recv_size[translate()(ii, jj, kk)] * max_fields_n * sizeof(DataType),
                                    ii_P,
                                    jj_P,
                                    kk_P);
                            } else {
                                send_size[translate()(ii, jj, kk)] = 0;
                                send_buffer[translate()(ii, jj, kk)] = nullptr;

                                base_type::m_haloexch.register_send_to_buffer(nullptr, 0, ii_P, jj_P, kk_P);

                                recv_size[translate()(ii, jj, kk)] = 0;

                                recv_buffer[translate()(ii, jj, kk)] = nullptr;

                                base_type::m_haloexch.register_receive_from_buffer(nullptr, 0, ii_P, jj_P, kk_P);
                            }
                        }

            GT_CUDA_CHECK(cudaMalloc((&d_send_buffer), _impl::static_pow3<DIMS>::value * sizeof(DataType *)));

            GT_CUDA_CHECK(cudaMemcpy(d_send_buffer,
                &(send_buffer[0]),
                _impl::static_pow3<DIMS>::value * sizeof(DataType *),
                cudaMemcpyHostToDevice));

            GT_CUDA_CHECK(cudaMalloc(&d_recv_buffer, _impl::static_pow3<DIMS>::value * sizeof(DataType *)));

            GT_CUDA_CHECK(cudaMemcpy(d_recv_buffer,
                &(recv_buffer[0]),
                _impl::static_pow3<DIMS>::value * sizeof(DataType *),
                cudaMemcpyHostToDevice));

            GT_CUDA_CHECK(cudaMalloc(&d_send_size, _impl::static_pow3<DIMS>::value * sizeof(int)));

            GT_CUDA_CHECK(cudaMemcpy(
                d_send_size, &(send_size[0]), _impl::static_pow3<DIMS>::value * sizeof(int), cudaMemcpyHostToDevice));

            GT_CUDA_CHECK(cudaMalloc(&d_recv_size, _impl::static_pow3<DIMS>::value * sizeof(int)));

            GT_CUDA_CHECK(cudaMemcpy(
                d_recv_size, &(recv_size[0]), _impl::static_pow3<DIMS>::value * sizeof(int), cudaMemcpyHostToDevice));

            GT_CUDA_CHECK(cudaMalloc(&halo_d, DIMS * sizeof(halo_descriptor)));

            GT_CUDA_CHECK(cudaMemcpy(
                halo_d, dangeroushalo /*halo.raw_array()*/, DIMS * sizeof(halo_descriptor), cudaMemcpyHostToDevice));

            GT_CUDA_CHECK(cudaMalloc(&halo_d_r, DIMS * sizeof(halo_descriptor)));

            GT_CUDA_CHECK(cudaMemcpy(halo_d_r,
                dangeroushalo_r /*halo.raw_array()*/,
                DIMS * sizeof(halo_descriptor),
                cudaMemcpyHostToDevice));
        }

        /**
           Function to pack data before sending

           \param[in] fields vector with data fields pointers to be packed from
        */
        template <typename... Pointers>
        void pack(const Pointers *... fields) {
            typedef translate_t<3, default_layout_map<3>::type> translate;
            auto ints = meta::make_integer_sequence<unsigned int, sizeof...(Pointers)>{};
            if (send_size[translate()(0, 0, -1)]) {
                m_packZL_variadic(d_send_buffer, d_send_size, dangeroushalo, halo_d, std::make_tuple(fields...), ints);
            }

            if (send_size[translate()(0, 0, 1)]) {
                m_packZU_variadic(d_send_buffer, d_send_size, dangeroushalo, halo_d, std::make_tuple(fields...), ints);
            }

            if (send_size[translate()(0, -1, 0)]) {
                m_packYL_variadic(d_send_buffer, d_send_size, dangeroushalo, halo_d, std::make_tuple(fields...), ints);
            }

            if (send_size[translate()(0, 1, 0)]) {
                m_packYU_variadic(d_send_buffer, d_send_size, dangeroushalo, halo_d, std::make_tuple(fields...), ints);
            }

            if (send_size[translate()(-1, 0, 0)]) {
                m_packXL_variadic(d_send_buffer, d_send_size, dangeroushalo, halo_d, std::make_tuple(fields...), ints);
            }

            if (send_size[translate()(1, 0, 0)]) {
                m_packXU_variadic(d_send_buffer, d_send_size, dangeroushalo, halo_d, std::make_tuple(fields...), ints);
            }

            for (int ii = -1; ii <= 1; ++ii)
                for (int jj = -1; jj <= 1; ++jj)
                    for (int kk = -1; kk <= 1; ++kk)
                        if (ii != 0 || jj != 0 || kk != 0) {
                            using translate_P = translate_t<3, proc_layout>;
                            using map_type = typename translate_P::map_type;
                            const int ii_P = pack_get_elem<map_type::template at<0>()>::apply(ii, jj, kk);
                            const int jj_P = pack_get_elem<map_type::template at<1>()>::apply(ii, jj, kk);
                            const int kk_P = pack_get_elem<map_type::template at<2>()>::apply(ii, jj, kk);

                            if (base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1) {
                                base_type::m_haloexch.set_send_to_size(
                                    send_size[translate()(ii, jj, kk)] * sizeof...(fields) * sizeof(DataType),
                                    ii_P,
                                    jj_P,
                                    kk_P);
                                base_type::m_haloexch.set_receive_from_size(
                                    recv_size[translate()(ii, jj, kk)] * sizeof...(fields) * sizeof(DataType),
                                    ii_P,
                                    jj_P,
                                    kk_P);
                            }
                        }

#ifdef GCL_MULTI_STREAMS
            GT_CUDA_CHECK(cudaStreamSynchronize(ZL_stream));
            GT_CUDA_CHECK(cudaStreamSynchronize(ZU_stream));
            GT_CUDA_CHECK(cudaStreamSynchronize(YL_stream));
            GT_CUDA_CHECK(cudaStreamSynchronize(YU_stream));
            GT_CUDA_CHECK(cudaStreamSynchronize(XL_stream));
            GT_CUDA_CHECK(cudaStreamSynchronize(XU_stream));
#else
            GT_CUDA_CHECK(cudaDeviceSynchronize());
#endif
        }

        template <typename... Pointers>
        void unpack(Pointers *... fields) {
            auto ints = meta::make_integer_sequence<unsigned int, sizeof...(Pointers)>{};
            typedef translate_t<3, default_layout_map<3>::type> translate;
            if (recv_size[translate()(0, 0, -1)]) {
                m_unpackZL_variadic(
                    d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r, std::make_tuple(fields...), ints);
            }
            if (recv_size[translate()(0, 0, 1)]) {
                m_unpackZU_variadic(
                    d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r, std::make_tuple(fields...), ints);
            }
            if (recv_size[translate()(0, -1, 0)]) {
                m_unpackYL_variadic(
                    d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r, std::make_tuple(fields...), ints);
            }
            if (recv_size[translate()(0, 1, 0)]) {
                m_unpackYU_variadic(
                    d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r, std::make_tuple(fields...), ints);
            }
            if (recv_size[translate()(-1, 0, 0)]) {
                m_unpackXL_variadic(
                    d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r, std::make_tuple(fields...), ints);
            }
            if (recv_size[translate()(1, 0, 0)]) {
                m_unpackXU_variadic(
                    d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r, std::make_tuple(fields...), ints);
            }
        }

        /**
           Function to pack data before sending

           \param[in] fields vector with data fields pointers to be packed from
        */
        void pack(std::vector<DataType *> const &fields) {
            typedef translate_t<3, default_layout_map<3>::type> translate;
            if (send_size[translate()(0, 0, -1)]) {
                m_packZL(fields, d_send_buffer, d_send_size, dangeroushalo, halo_d);
            }
            if (send_size[translate()(0, 0, 1)]) {
                m_packZU(fields, d_send_buffer, d_send_size, dangeroushalo, halo_d);
            }
            if (send_size[translate()(0, -1, 0)]) {
                m_packYL(fields, d_send_buffer, d_send_size, dangeroushalo, halo_d);
            }
            if (send_size[translate()(0, 1, 0)]) {
                m_packYU(fields, d_send_buffer, d_send_size, dangeroushalo, halo_d);
            }
            if (send_size[translate()(-1, 0, 0)]) {
                m_packXL(fields, d_send_buffer, d_send_size, dangeroushalo, halo_d);
            }
            if (send_size[translate()(1, 0, 0)]) {
                m_packXU(fields, d_send_buffer, d_send_size, dangeroushalo, halo_d);
            }

            for (int ii = -1; ii <= 1; ++ii)
                for (int jj = -1; jj <= 1; ++jj)
                    for (int kk = -1; kk <= 1; ++kk)
                        if (ii != 0 || jj != 0 || kk != 0) {
                            using translate_P = translate_t<3, proc_layout>;
                            using map_type = typename translate_P::map_type;
                            const int ii_P = pack_get_elem<map_type::template at<0>()>::apply(ii, jj, kk);
                            const int jj_P = pack_get_elem<map_type::template at<1>()>::apply(ii, jj, kk);
                            const int kk_P = pack_get_elem<map_type::template at<2>()>::apply(ii, jj, kk);

                            if (base_type::pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1) {
                                base_type::m_haloexch.set_send_to_size(
                                    send_size[translate()(ii, jj, kk)] * fields.size() * sizeof(DataType),
                                    ii_P,
                                    jj_P,
                                    kk_P);
                                base_type::m_haloexch.set_receive_from_size(
                                    recv_size[translate()(ii, jj, kk)] * fields.size() * sizeof(DataType),
                                    ii_P,
                                    jj_P,
                                    kk_P);
                            }
                        }

// perform device syncronization to ensure that packing is finished
// before MPI is called with the device pointers, otherwise stale
// information can be sent
#ifdef GCL_MULTI_STREAMS
            GT_CUDA_CHECK(cudaStreamSynchronize(ZL_stream));
            GT_CUDA_CHECK(cudaStreamSynchronize(ZU_stream));
            GT_CUDA_CHECK(cudaStreamSynchronize(YL_stream));
            GT_CUDA_CHECK(cudaStreamSynchronize(YU_stream));
            GT_CUDA_CHECK(cudaStreamSynchronize(XL_stream));
            GT_CUDA_CHECK(cudaStreamSynchronize(XU_stream));
#else
            GT_CUDA_CHECK(cudaDeviceSynchronize());
#endif
            // CODE TO SEE THE PACKED BUFFERS
            //     double *send_buffer_l[27];
            // for(int k=0; k<3; k++)
            //   for(int j=0; j<3; j++)
            //     for(int i=0; i<3; i++)
            //       if (i!=1 || j!=1 || k!=1) {
            //         send_buffer_l[k*9 + i+3*j] = (double*)malloc(send_size[k*9 + i+3*j]*sizeof(double)*3);
            //       }

            //     int size, err;
            // for(int k=0; k<3; k++)
            //   //if (k==0)// || k==2)
            //   for(int j=0; j<3; j++)
            //     for(int i=0; i<3; i++)
            //       if (i!=1 || j!=1 || k!=1) {
            //         size = send_size[k*9 + i+3*j] * sizeof(double)*3;
            //         printf("START transferring bufer i=%d, j=%d, size=%d\n",
            //                  i, j, size);
            //         err = cudaMemcpy(send_buffer_l[k*9 + i+3*j], send_buffer[k*9 + i+3*j], size,
            //                        cudaMemcpyDeviceToHost);
            //         if(err != cudaSuccess) {
            //           printf("Error transferring bufer i=%d, j=%d, size=%d\n",
            //                  i, j, size);
            //           exit(-1);
            //         }
            //         printf("OK transferring bufer i=%d, j=%d, size=%d\n",
            //                  i, j, size);
            //       }

            // printf("printing buffers\n");
            // // exit(0);

            // // display content of all the buffers
            // for(int k=0; k<3; k++)
            //   //    if (k==0)// || k==2)
            //   for(int j=0; j<3; j++)
            //     for(int i=0; i<3; i++)
            //       if (i!=1 || j!=1 || k!=1) {
            //         printf("Buffer = %d\n", j*3 + i);
            //         for(int l=0; l < dangeroushalo[2].s_length(k-1); l++)
            //           for(int m=0; m < dangeroushalo[1].s_length(j-1); m++)
            //             for(int n=0; n < dangeroushalo[0].s_length(i-1); n++){
            //               int ind = l * dangeroushalo[1].s_length(j-1) * dangeroushalo[0].s_length(i-1) + m *
            //               dangeroushalo[0].s_length(i-1) + n;
            //               long long int value = static_cast<long long int>(send_buffer_l[9*k + j*3+i][ind]);
            //               printf("PACKED %d; %d / %d / %d (%d) = %lld %lld %lld \n",
            //                      9*k+j*3+i, n, m, l, ind, value%10000, (value/10000)%10000, (value/100000000)%10000);
            //             }
            //       }
        }

        /**
           Function to unpack received data

           \param[in] fields vector with data fields pointers to be unpacked into
        */
        void unpack(std::vector<DataType *> const &fields) {
            typedef translate_t<3, default_layout_map<3>::type> translate;
            if (recv_size[translate()(0, 0, -1)]) {
                m_unpackZL(fields, d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r);
            }
            if (recv_size[translate()(0, 0, 1)]) {
                m_unpackZU(fields, d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r);
            }
            if (recv_size[translate()(0, -1, 0)]) {
                m_unpackYL(fields, d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r);
            }
            if (recv_size[translate()(0, 1, 0)]) {
                m_unpackYU(fields, d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r);
            }
            if (recv_size[translate()(-1, 0, 0)]) {
                m_unpackXL(fields, d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r);
            }
            if (recv_size[translate()(1, 0, 0)]) {
                m_unpackXU(fields, d_recv_buffer, d_recv_size, dangeroushalo_r, halo_d_r);
            }
        }
    };
#endif
} // namespace gridtools
