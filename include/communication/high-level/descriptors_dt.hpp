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
#ifndef _DESCRIPTORS_DT_H_
#define _DESCRIPTORS_DT_H_

#include "../../common/array.hpp"
#include <vector>
#include "../low-level/proc_grids_3D.hpp"
#include "../low-level/Halo_Exchange_3D.hpp"
#include "../../common/make_array.hpp"
#include <common/gt_assert.hpp>
#include "../../common/boollist.hpp"
#include "../../common/ndloops.hpp"
#include "../low-level/data_types_mapping.hpp"
#include "gcl_parameters.hpp"
#include "../../common/numerics.hpp"
#include "empty_field_base.hpp"
#include "../../common/layout_map.hpp"

#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include "descriptors_fwd.hpp"
#include "descriptor_base.hpp"
#include "helpers_impl.hpp"
#include <boost/type_traits/remove_pointer.hpp>
#include "access.hpp"

namespace gridtools {

    /** \class empty_field
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

        \tparam DataType Type of elements contained in data arrays
        \tparam DIMS the number of dimensions of the data field

    */
    template < typename DataType, int DIMS >
    class empty_field : public empty_field_base< DataType, DIMS > {
        typedef empty_field_base< DataType, DIMS > base_type;

      public:
        /**
            Constructor that receive the pointer to the data. This is explicit and
            must then be called.
        */
        explicit empty_field() {}

        const halo_descriptor *raw_array() const { return &(base_type::halos[0]); }

        /** void pack(gridtools::array<int, D> const& eta, iterator &it)
            Pack the elements of a data field passed in input as iterator_in to be sent using the
            iterator_out passed in that points to data buffers. At the end
            the iterator_out points to the element next to the last inserted. In inout
            the iterator_out points to the elements to be insered

            \param[in] eta the eta parameter as indicated in \link MULTI_DIM_ACCESS \endlink
            \param[in] field_ptr iterator pointing to data field data
            \param[in,out] it iterator pointing to the data.
        */
        template < typename iterator_in, typename iterator_out >
        void pack(gridtools::array< int, DIMS > const &eta, iterator_in const &field_ptr, iterator_out &it) const {
            if (base_type::MPDT_INSIDE[_impl::neigh_idx(eta)].second) {
                int ss2;
                MPI_Pack_size(1, base_type::MPDT_INSIDE[_impl::neigh_idx(eta)].first, gridtools::GCL_WORLD, &ss2);

                int I = 0;
                MPI_Pack(field_ptr,
                    1,
                    base_type::MPDT_INSIDE[_impl::neigh_idx(eta)].first,
                    it,
                    ss2,
                    &I,
                    gridtools::GCL_WORLD);

                it +=
                    I /
                    sizeof(
                        typename boost::remove_pointer< typename boost::remove_reference< iterator_out >::type >::type);
            } else {
                // nothing here
            }
        }

        /** void unpack(gridtools::array<int, D> const& eta, iterator &it)
            Unpack the elements into a data field passed in input as
            iterator_in that have being received in data obtained by the
            iterator_out passed in that points to data buffers. At the end
            the iterator points to the element next to the last read element. In inout
            the iterator points to the elements to be extracted from buffers and put
            int the halo region.

            \param[in] eta the eta parameter as explained in \link MULTI_DIM_ACCESS \endlink of the sending neighbor
            \param[in] field_ptr iterator pointing to data field data
            \param[in,out] it iterator pointing to the data in buffers.
        */
        template < typename iterator_in, typename iterator_out >
        void unpack(gridtools::array< int, DIMS > const &eta, iterator_in const &field_ptr, iterator_out &it) const {
            if (base_type::MPDT_OUTSIDE[_impl::neigh_idx(eta)].second) {
                int I = 0;

                MPI_Unpack(it,
                    base_type::recv_buffer_size(eta) * sizeof(DataType),
                    &I,
                    field_ptr,
                    1,
                    base_type::MPDT_OUTSIDE[_impl::neigh_idx(eta)].first,
                    gridtools::GCL_WORLD);

                it +=
                    I /
                    sizeof(
                        typename boost::remove_pointer< typename boost::remove_reference< iterator_out >::type >::type);
            } else {
                // nothing here
            }
        }

        template < typename iterator >
        void pack_all(gridtools::array< int, DIMS > const &, iterator &it) const {}

/**
   This method takes a tuple eta identifiyng a neighbor \link MULTI_DIM_ACCESS \endlink
   and a list of data fields and pack all the data corresponding
   to the halo described by the class. The data is packed starting at
   position pointed by iterator and the iterator will point to the next free
   position at the end of the operation.

   \param[in] eta the eta parameter as explained in \link MULTI_DIM_ACCESS \endlink of the receiving neighbor
   \param[in,out] it iterator pointing to  storage area where data is packed
   \param[in] field the first data field to be processed
   \param[in] args the rest of the list of data fields to be packed (they may have different datatypes).
 */
#ifdef CXX11_ENABLED
        template < typename iterator, typename FIRST, typename... FIELDS >
        void pack_all(
            gridtools::array< int, DIMS > const &eta, iterator &it, FIRST const &field, const FIELDS &... args) const {
            pack(eta, field, it);
            pack_all(eta, it, args...);
        }
#else
#define MACRO_IMPL(z, n, _)                                                                    \
    template < typename iterator, BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(n), typename FIELD) > \
    void pack_all(gridtools::array< int, DIMS > const &eta,                                    \
        iterator &it,                                                                          \
        BOOST_PP_ENUM_BINARY_PARAMS_Z(z, BOOST_PP_INC(n), FIELD, const &arg)) const {          \
        pack_all(eta, it BOOST_PP_COMMA_IF(n) BOOST_PP_ENUM_PARAMS_Z(z, n, arg));              \
        pack(eta, arg##n, it);                                                                 \
    }

        BOOST_PP_REPEAT(GCL_MAX_FIELDS, MACRO_IMPL, all)
#undef MACRO_IMPL
#endif

        template < typename iterator >
        void unpack_all(gridtools::array< int, DIMS > const &, iterator &it) const {}

/**
   This method takes a tuple eta identifiyng a neighbor \link MULTI_DIM_ACCESS \endlink
   and a list of data fields and pack all the data corresponding
   to the halo described by the class. The data is packed starting at
   position pointed by iterator and the iterator will point to the next free
   position at the end of the operation.

   \param[in] eta the eta parameter as explained in \link MULTI_DIM_ACCESS \endlink of the sending neighbor
   \param[in,out] it iterator pointing to the data to be unpacked
   \param[in] field the first data field to be processed
   \param[in] args the rest of the list of data fields where data has to be unpacked into (they may have different
   datatypes).
 */
#ifdef CXX11_ENABLED
        template < typename iterator, typename FIRST, typename... FIELDS >
        void unpack_all(
            gridtools::array< int, DIMS > const &eta, iterator &it, FIRST const &field, const FIELDS &... args) const {
            unpack(eta, field, it);
            unpack_all(eta, it, args...);
        }
#else
#define MACRO_IMPL(z, n, _)                                                                    \
    template < typename iterator, BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(n), typename FIELD) > \
    void unpack_all(gridtools::array< int, DIMS > const &eta,                                  \
        iterator &it,                                                                          \
        BOOST_PP_ENUM_BINARY_PARAMS_Z(z, BOOST_PP_INC(n), FIELD, const &arg)) const {          \
        unpack_all(eta, it BOOST_PP_COMMA_IF(n) BOOST_PP_ENUM_PARAMS_Z(z, n, arg));            \
        unpack(eta, arg##n, it);                                                               \
    }

        BOOST_PP_REPEAT(GCL_MAX_FIELDS, MACRO_IMPL, all)
#undef MACRO_IMPL
#endif
    };

    template < typename T, int I >
    std::ostream &operator<<(std::ostream &s, empty_field< T, I > const &ef) {
        s << "empty_field ";
        for (int i = 0; i < I; ++i)
            s << ef.raw_array()[i] << ", ";
        return s;
    }

    /** \class field_descriptor
        Class containint the information about a data field (grid).
        It contains a pointer to the first element of the data field,
        the number of dimensions as a template argument and the size of the
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

        \tparam DataType type of lements of the datafield
        \tparam DIMS the number of dimensions of the data field
    */
    template < typename DataType, int DIMS >
    class field_descriptor : public empty_field< DataType, DIMS > {
        DataType *fieldptr; // Pointer to the data field

        typedef empty_field< DataType, DIMS > base_type;

      public:
        /**
            Constructor that receive the pointer to the data. This is explicit and
            must then be called.
            \param[in] _fp DataType* pointer to the data field
        */
        explicit field_descriptor(DataType *_fp) : fieldptr(_fp) {}

        /** void pack(gridtools::array<int, D> const& eta, iterator &it)
            Pack the elements to be sent using the iterator passed in. At the end
            the iterator points to the element next to the last inserted. In inout
            the iterator points to the elements to be insered

            \param[in] eta the eta parameter as indicated in \link MULTI_DIM_ACCESS \endlink
            \param[in,out] it iterator pointing to the data.
        */
        template < typename iterator >
        void pack(gridtools::array< int, DIMS > const &eta, iterator &it) const {
            base_type::pack(eta, fieldptr, it);
        }

        /** void unpack(gridtools::array<int, D> const& eta, iterator &it)
            Unpack the elements received using the iterator passed in.. At the end
            the iterator points to the element next to the last read element. In inout
            the iterator points to the elements to be extracted from buffers and put
            int the halo region.

            \param[in] eta the eta parameter as explained in \link MULTI_DIM_ACCESS \endlink of the sending neighbor
            \param[in,out] it iterator pointing to the data in buffers.
        */
        template < typename iterator >
        void unpack(gridtools::array< int, DIMS > const &eta, iterator &it) const {
            base_type::unpack(eta, fieldptr, it);
        }
    };

    /**
        Class containing the description of one halo and a communication
        pattern.  A communication is triggered when a list of data
        fields are passed to the exchange functions, when the data
        according to the halo descriptors are echanged. This class is
        needed when the addresses and the number of the data fields
        changes dynamically but the sizes are constant. Data elements
        for each hndlr_dynamic_ut must be the same.

        \tparam DataType Type of the elements in data arrays
        \tparam DIMS Number of dimensions of the grids.
        \tparam HaloExch Communication patter with halo exchange.
        \tparam proc_layout Map between dimensions in increasing-stride order and processor grid dimensions
        \tparam Gcl_Arch Specification of architecture used to indicate where the data is L3/include/gcl_arch.h file
       reference
    */
    template < typename DataType,
        typename GridType,
        typename HaloExch,
        typename proc_layout = typename default_layout_map< GridType::ndims >::type,
        typename Gcl_Arch = gridtools::gcl_cpu,
        int VERSION = 0 >
    class hndlr_dynamic_ut : public descriptor_base< HaloExch > {
        typedef hndlr_dynamic_ut< DataType, GridType, HaloExch, proc_layout, Gcl_Arch, VERSION > this_type;
        static const int DIMS = GridType::ndims;

      public:
        empty_field< DataType, DIMS > halo;

      private:
        gridtools::array< DataType *, _impl::static_pow3< DIMS >::value > send_buffer; // One entry will not be used...
        gridtools::array< DataType *, _impl::static_pow3< DIMS >::value > recv_buffer;

      public:
        typedef descriptor_base< HaloExch > base_type;
        typedef base_type pattern_type;

        /** Architecture type
         */
        typedef Gcl_Arch arch_type;

        /**
           Type of the computin grid associated to the pattern
         */
        typedef typename pattern_type::grid_type grid_type;

        /**
           Type of the translation used to map dimensions to buffer addresses
         */
        typedef translate_t< DIMS, typename default_layout_map< DIMS >::type > translate;

      private:
        hndlr_dynamic_ut(hndlr_dynamic_ut const &) {}

      public:
#ifdef GCL_TRACE
        void set_pattern_tag(int tag) { base_type::m_haloexch.set_pattern_tag(tag); };
#endif

        /**
           Constructor

           \param[in] c The object of the class used to specify periodicity in each dimension
           \param[in] comm MPI communicator (typically MPI_Comm_world)
        */
        template < typename Array >
        explicit hndlr_dynamic_ut(typename grid_type::period_type const &c, MPI_Comm comm, Array const *dimensions)
            : base_type(c, comm, dimensions), halo() {}

        ~hndlr_dynamic_ut() {
#ifdef _GCL_CHECK_DESTRUCTOR
            std::cout << "Destructor " << __FILE__ << ":" << __LINE__ << std::endl;
#endif

            destroy< DIMS, 0 >::doit(*this);
        }

        template < int D, int N >
        struct destroy;

        template < int N >
        struct destroy< 3, N > {
            template < typename T >
            static void doit(T &descriptor) {
                for (int i = -1; i <= 1; ++i)
                    for (int j = -1; j <= 1; ++j)
                        for (int k = -1; k <= 1; ++k) {
                            if (!descriptor.send_buffer[translate()(i, j, k)])
                                _impl::gcl_alloc< DataType, arch_type >::free(
                                    descriptor.send_buffer[translate()(i, j, k)]);
                            if (!descriptor.recv_buffer[translate()(i, j, k)])
                                _impl::gcl_alloc< DataType, arch_type >::free(
                                    descriptor.recv_buffer[translate()(i, j, k)]);
                        }
            }
        };

        /**
           Constructor

           \param[in] c The object of the class used to specify periodicity in each dimension
           \param[in] _P Number of processors the pattern is running on (numbered from 0 to _P-1
           \param[in] _pid Integer identifier of the process calling the constructor
         */
        explicit hndlr_dynamic_ut(typename grid_type::period_type const &c, int _P, int _pid)
            : base_type(grid_type(c, _P, _pid)), halo() {}

        /**
           Constructor

           \param[in] g A processor grid that will execute the pattern
         */
        explicit hndlr_dynamic_ut(grid_type const &g) : halo(), base_type(g) {}

        /**
           Function to setup internal data structures for data exchange and preparing eventual underlying layers

           The use of this function is deprecated

           \param max_fields_n Maximum number of data fields that will be passed to the communication functions
        */
        void allocate_buffers(int max_fields_n) { setup(max_fields_n); }

        /**
           Function to setup internal data structures for data exchange and preparing eventual underlying layers

           \param max_fields_n Maximum number of data fields that will be passed to the communication functions
        */
        void setup(int max_fields_n) {
            halo.setup();
            _impl::allocation_service< this_type >()(this, max_fields_n);
        }

/**
   Function to pack data to be sent

   \param[in] _fields data fields to be packed
*/
#ifdef CXX11_ENABLED
        template < typename... FIELDS >
        void pack(const FIELDS &... _fields) const {
            pack_dims< DIMS, 0 >()(*this, _fields...);
        }
#else
#define MACRO_IMPL(z, n, _)                                                                    \
    template < BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(n), typename FIELD) >                    \
    void pack(BOOST_PP_ENUM_BINARY_PARAMS_Z(z, BOOST_PP_INC(n), FIELD, const &_field)) const { \
        pack_dims< DIMS, 0 >()(*this, BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(n), _field));     \
    }

        BOOST_PP_REPEAT(GCL_MAX_FIELDS, MACRO_IMPL, all)
#undef MACRO_IMPL
#endif

/**
   Function to unpack received data

   \param[in] _fields data fields where to unpack data
*/
#ifdef CXX11_ENABLED
        template < typename... FIELDS >
        void unpack(const FIELDS &... _fields) const {
            unpack_dims< DIMS, 0 >()(*this, _fields...);
        }
#else
#define MACRO_IMPL(z, n, _)                                                                      \
    template < BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(n), typename FIELD) >                      \
    void unpack(BOOST_PP_ENUM_BINARY_PARAMS_Z(z, BOOST_PP_INC(n), FIELD, const &_field)) const { \
        unpack_dims< DIMS, 0 >()(*this, BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(n), _field));     \
    }

        BOOST_PP_REPEAT(GCL_MAX_FIELDS, MACRO_IMPL, all)
#undef MACRO_IMPL
#endif

        /**
           Function to unpack received data

           \param[in] fields vector with data fields pointers to be packed from
        */
        void pack(std::vector< DataType * > const &fields) { pack_vector_dims< DIMS, 0 >()(*this, fields); }

        /**
           Function to unpack received data

           \param[in] fields vector with data fields pointers to be unpacked into
        */
        void unpack(std::vector< DataType * > const &fields) { unpack_vector_dims< DIMS, 0 >()(*this, fields); }

        /// Utilities

        // FRIENDING
        friend struct _impl::allocation_service< this_type >;

      private:
        template < int I, int dummy >
        struct pack_dims {};

        template < int dummy >
        struct pack_dims< 3, dummy > {
#ifdef CXX11_ENABLED
            template < typename T, typename... FIELDS >
            void operator()(const T &hm, const FIELDS &... _fields) const {
#pragma omp parallel for schedule(dynamic, 1) collapse(3)
                for (int ii = -1; ii <= 1; ++ii) {
                    for (int jj = -1; jj <= 1; ++jj) {
                        for (int kk = -1; kk <= 1; ++kk) {
                            typedef proc_layout map_type;
                            const int ii_P = map_type().template select< 0 >(ii, jj, kk);
                            const int jj_P = map_type().template select< 1 >(ii, jj, kk);
                            const int kk_P = map_type().template select< 2 >(ii, jj, kk);
                            if ((ii != 0 || jj != 0 || kk != 0) &&
                                (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                                DataType *it = &(hm.send_buffer[translate()(ii, jj, kk)][0]);
                                hm.halo.pack_all(make_array(ii, jj, kk), it, _fields...);
                            }
                        }
                    }
                }
            }
#else

#ifndef _GCL_GPU_
#define PUT_OMP _Pragma("omp parallel for schedule(dynamic) collapse(3)")
#else
#define PUT_OMP
#endif

#define MACRO_IMPL(z, n, _)                                                                                           \
    template < typename T, BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(n), typename FIELD) >                               \
    void operator()(const T &hm, BOOST_PP_ENUM_BINARY_PARAMS_Z(z, BOOST_PP_INC(n), FIELD, const &_field)) const {     \
        PUT_OMP                                                                                                       \
        for (int ii = -1; ii <= 1; ++ii) {                                                                            \
            for (int jj = -1; jj <= 1; ++jj) {                                                                        \
                for (int kk = -1; kk <= 1; ++kk) {                                                                    \
                    typedef proc_layout map_type;                                                                     \
                    const int ii_P = map_type().template select< 0 >(ii, jj, kk);                                     \
                    const int jj_P = map_type().template select< 1 >(ii, jj, kk);                                     \
                    const int kk_P = map_type().template select< 2 >(ii, jj, kk);                                     \
                    if ((ii != 0 || jj != 0 || kk != 0) && (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) { \
                        DataType *it = &(hm.send_buffer[translate()(ii, jj, kk)][0]);                                 \
                        hm.halo.pack_all(                                                                             \
                            make_array(ii, jj, kk), it, BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(n), _field));          \
                    }                                                                                                 \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
    }

            BOOST_PP_REPEAT(GCL_MAX_FIELDS, MACRO_IMPL, all)
#undef MACRO_IMPL
#undef PUT_OMP
#endif
        };

        template < int I, int dummy >
        struct unpack_dims {};

        template < int dummy >
        struct unpack_dims< 3, dummy > {
#ifdef CXX11_ENABLED
            template < typename T, typename... FIELDS >
            void operator()(const T &hm, const FIELDS &... _fields) const {
#pragma omp parallel for schedule(dynamic, 1) collapse(3)
                for (int ii = -1; ii <= 1; ++ii) {
                    for (int jj = -1; jj <= 1; ++jj) {
                        for (int kk = -1; kk <= 1; ++kk) {
                            typedef proc_layout map_type;
                            const int ii_P = map_type().template select< 0 >(ii, jj, kk);
                            const int jj_P = map_type().template select< 1 >(ii, jj, kk);
                            const int kk_P = map_type().template select< 2 >(ii, jj, kk);
                            if ((ii != 0 || jj != 0 || kk != 0) &&
                                (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                                DataType *it = &(hm.recv_buffer[translate()(ii, jj, kk)][0]);
                                hm.halo.unpack_all(make_array(ii, jj, kk), it, _fields...);
                            }
                        }
                    }
                }
            }
#else

#ifndef _GCL_GPU_
#define PUT_OMP _Pragma("omp parallel for schedule(dynamic) collapse(3)")
#else
#define PUT_OMP
#endif

#define MACRO_IMPL(z, n, _)                                                                                           \
    template < typename T, BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(n), typename FIELD) >                               \
    void operator()(const T &hm, BOOST_PP_ENUM_BINARY_PARAMS_Z(z, BOOST_PP_INC(n), FIELD, const &_field)) const {     \
        PUT_OMP                                                                                                       \
        for (int ii = -1; ii <= 1; ++ii) {                                                                            \
            for (int jj = -1; jj <= 1; ++jj) {                                                                        \
                for (int kk = -1; kk <= 1; ++kk) {                                                                    \
                    typedef proc_layout map_type;                                                                     \
                    const int ii_P = map_type().template select< 0 >(ii, jj, kk);                                     \
                    const int jj_P = map_type().template select< 1 >(ii, jj, kk);                                     \
                    const int kk_P = map_type().template select< 2 >(ii, jj, kk);                                     \
                    if ((ii != 0 || jj != 0 || kk != 0) && (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) { \
                        DataType *it = &(hm.recv_buffer[translate()(ii, jj, kk)][0]);                                 \
                        hm.halo.unpack_all(                                                                           \
                            make_array(ii, jj, kk), it, BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_INC(n), _field));          \
                    }                                                                                                 \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
    }

            BOOST_PP_REPEAT(GCL_MAX_FIELDS, MACRO_IMPL, all)
#undef MACRO_IMPL
#undef PUT_OMP
#endif
        };

        template < int I, int dummy >
        struct pack_vector_dims {};

        template < int dummy >
        struct pack_vector_dims< 3, dummy > {
            template < typename T >
            void operator()(const T &hm, std::vector< DataType * > const &fields) const {
#pragma omp parallel for schedule(dynamic, 1) collapse(3)
                for (int ii = -1; ii <= 1; ++ii) {
                    for (int jj = -1; jj <= 1; ++jj) {
                        for (int kk = -1; kk <= 1; ++kk) {
                            typedef proc_layout map_type;
                            const int ii_P = map_type().template select< 0 >(ii, jj, kk);
                            const int jj_P = map_type().template select< 1 >(ii, jj, kk);
                            const int kk_P = map_type().template select< 2 >(ii, jj, kk);
                            if ((ii != 0 || jj != 0 || kk != 0) &&
                                (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                                DataType *it = &(hm.send_buffer[translate()(ii, jj, kk)][0]);
                                for (size_t i = 0; i < fields.size(); ++i) {
                                    hm.halo.pack(make_array(ii, jj, kk), fields[i], it);
                                }
                            }
                        }
                    }
                }
            }
        };

        template < int I, int dummy >
        struct unpack_vector_dims {};

        template < int dummy >
        struct unpack_vector_dims< 3, dummy > {
            template < typename T >
            void operator()(const T &hm, std::vector< DataType * > const &fields) const {
#pragma omp parallel for schedule(dynamic, 1) collapse(3)
                for (int ii = -1; ii <= 1; ++ii) {
                    for (int jj = -1; jj <= 1; ++jj) {
                        for (int kk = -1; kk <= 1; ++kk) {
                            typedef proc_layout map_type;
                            const int ii_P = map_type().template select< 0 >(ii, jj, kk);
                            const int jj_P = map_type().template select< 1 >(ii, jj, kk);
                            const int kk_P = map_type().template select< 2 >(ii, jj, kk);
                            if ((ii != 0 || jj != 0 || kk != 0) &&
                                (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                                DataType *it = &(hm.recv_buffer[translate()(ii, jj, kk)][0]);
                                for (size_t i = 0; i < fields.size(); ++i) {
                                    hm.halo.unpack(make_array(ii, jj, kk), fields[i], it);
                                }
                            }
                        }
                    }
                }
            }
        };
    };

    /** \class handler_manager_ut
        Handler Manager is a class that keeps ona hndlr_descriptor and provide the
        handlers for the library, the size information, the buffers allocation, and the
        exchange calls.

        \tparam DataType type of the elements of the data fields associated to the handler.
        \tparam DIMS Number of dimensions of the grids.
        \tparam HaloExch pattern type used in communication.
    */
    template < typename DataType, int DIMS, typename HaloExch >
    class handler_manager_ut {
        hndlr_descriptor_ut< DataType, DIMS, HaloExch > *h;

      public:
        /**
           Creates a handler descriptor with a given argument.

           \tparam ARG Type of the input argument
           \param[in] c The object of the class used to specify periodicity in each dimension
           \param arg  Value of the argument
        */
        template < typename ARG >
        hndlr_descriptor_ut< DataType, DIMS, HaloExch > &create_handler(
            typename HaloExch::grid_type::period_type const &c, ARG const &arg) {
            h = new hndlr_descriptor_ut< DataType, DIMS, HaloExch >(c, arg);
            return *h;
        }

        /**
           Creates a handler descriptor with a given two arguments.

           \tparam ARG0 Type of the input argument
           \tparam ARG1 Type of the input argument
           \param[in] c The object of the class used to specify periodicity in each dimension
           \param arg0  Value of the argument
           \param arg1  Value of the argument
        */
        template < typename ARG0, typename ARG1 >
        hndlr_descriptor_ut< DataType, DIMS, HaloExch > &create_handler(
            typename HaloExch::grid_type::period_type const &c, ARG0 const &arg0, ARG1 const &arg1) {
            h = new hndlr_descriptor_ut< DataType, DIMS, HaloExch >(c, arg0, arg1);
            return *h;
        }

        /**
           Destroy the handler created by create_handler which cannot be reused
           after this function returns.
        */
        void destroy_handler(hndlr_descriptor_ut< DataType, DIMS, HaloExch > &h) { delete (&h); }
    };

    template < typename HaloExch, typename proc_layout_abs, typename Gcl_Arch, int version >
    class hndlr_generic< 3, HaloExch, proc_layout_abs, Gcl_Arch, version > : public descriptor_base< HaloExch > {
        static const int DIMS = 3;
        gridtools::array< char *, _impl::static_pow3< DIMS >::value > send_buffer; // One entry will not be used...
        gridtools::array< char *, _impl::static_pow3< DIMS >::value > recv_buffer;
        gridtools::array< int, _impl::static_pow3< DIMS >::value > send_buffer_size; // One entry will not be used...
        gridtools::array< int, _impl::static_pow3< DIMS >::value > recv_buffer_size;

        typedef Gcl_Arch arch_type;

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

            for (int i = -1; i <= 1; ++i)
                for (int j = -1; j <= 1; ++j)
                    for (int k = -1; k <= 1; ++k) {
                        if (!send_buffer[translate()(i, j, k)])
                            _impl::gcl_alloc< char, arch_type >::free(send_buffer[translate()(i, j, k)]);
                        if (!recv_buffer[translate()(i, j, k)])
                            _impl::gcl_alloc< char, arch_type >::free(recv_buffer[translate()(i, j, k)]);
                    }
        }

        /**
           Setup function, in this version, takes tree parameters to
           compute internal buffers and sizes. It takes a field on the fly
           struct, which requires Datatype and layout map template
           arguments that are inferred, so the user is not aware of them.

           \tparam DataType This type is inferred by halo_example paramter
           \tparam layomap This type is inferred by halo_example paramter

           \param[in] max_fields_n Maximum number of grids used in a computation
           \param[in] halo_example The (at least) maximal grid that is goinf to be used
           \param[in] typesize In case the DataType of the halo_example is not the same as the maximum data type used in
           the computation, this parameter can be given
         */
        template < typename DataType, typename _layomap, template < typename > class traits >
        void setup(int max_fields_n,
            field_on_the_fly< DataType, _layomap, traits > const &halo_example,
            int typesize = sizeof(DataType)) {
            typedef typename field_on_the_fly< DataType, _layomap, traits >::inner_layoutmap layomap;
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
                            send_buffer[translate()(i, j, k)] =
                                _impl::gcl_alloc< char, arch_type >::alloc(S * max_fields_n * typesize);
                            recv_buffer[translate()(i, j, k)] =
                                _impl::gcl_alloc< char, arch_type >::alloc(R * max_fields_n * typesize);
                            send_buffer_size[translate()(i, j, k)] = (S * max_fields_n * typesize);
                            recv_buffer_size[translate()(i, j, k)] = (R * max_fields_n * typesize);

                            // std::cout << halo_example << std::endl;
                            // printf("%d %d %d -> send %d, recv %d\n", i,j,k, send_buffer_size[translate()(i,j,k)],
                            // recv_buffer_size[translate()(i,j,k)]);

                            typedef typename layout_transform< layomap, proc_layout_abs >::type proc_layout;
                            const int i_P = proc_layout().template select< 0 >(i, j, k);
                            const int j_P = proc_layout().template select< 1 >(i, j, k);
                            const int k_P = proc_layout().template select< 2 >(i, j, k);

                            base_type::m_haloexch.register_send_to_buffer(
                                &(send_buffer[translate()(i, j, k)][0]), S * max_fields_n * typesize, i_P, j_P, k_P);

                            base_type::m_haloexch.register_receive_from_buffer(
                                &(recv_buffer[translate()(i, j, k)][0]), R * max_fields_n * typesize, i_P, j_P, k_P);
                        }
                    }
                }
            }
        }

        /**
           Setup function, in this version, takes a single parameter with
           an array of sizes to be associated with the halos.

           \tparam DataType This type is inferred by halo_example paramter
           \tparam layomap This type is inferred by halo_example paramter

           \param[in] buffer_size_list Array (gridtools::array) with the sizes of the buffers associated with the halos.
         */
        template < typename DataType, typename layomap >
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

                            typedef typename layout_transform< layomap, proc_layout_abs >::type proc_layout;
                            const int i_P = proc_layout().template select< 0 >(i, j, k);
                            const int j_P = proc_layout().template select< 1 >(i, j, k);
                            const int k_P = proc_layout().template select< 2 >(i, j, k);

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
                        char *it = reinterpret_cast< char * >(&(send_buffer[translate()(ii, jj, kk)][0]));
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
                        char *it = reinterpret_cast< char * >(&(recv_buffer[translate()(ii, jj, kk)][0]));
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
                const int ii_P = proc_layout().template select< 0 >(ii, jj, kk);
                const int jj_P = proc_layout().template select< 1 >(ii, jj, kk);
                const int kk_P = proc_layout().template select< 2 >(ii, jj, kk);
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
        const int ii_P = proc_layout().template select< 0 >(ii, jj, kk);                                          \
        const int jj_P = proc_layout().template select< 1 >(ii, jj, kk);                                          \
        const int kk_P = proc_layout().template select< 2 >(ii, jj, kk);                                          \
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
                const int ii_P = proc_layout().template select< 0 >(ii, jj, kk);
                const int jj_P = proc_layout().template select< 1 >(ii, jj, kk);
                const int kk_P = proc_layout().template select< 2 >(ii, jj, kk);
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
        const int ii_P = proc_layout().template select< 0 >(ii, jj, kk);                                          \
        const int jj_P = proc_layout().template select< 1 >(ii, jj, kk);                                          \
        const int kk_P = proc_layout().template select< 2 >(ii, jj, kk);                                          \
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
                const int ii_P = proc_layout().template select< 0 >(ii, jj, kk);
                const int jj_P = proc_layout().template select< 1 >(ii, jj, kk);
                const int kk_P = proc_layout().template select< 2 >(ii, jj, kk);
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
                const int ii_P = proc_layout().template select< 0 >(ii, jj, kk);
                const int jj_P = proc_layout().template select< 1 >(ii, jj, kk);
                const int kk_P = proc_layout().template select< 2 >(ii, jj, kk);
                if ((ii != 0 || jj != 0 || kk != 0) && (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                    for (unsigned int fi = 0; fi < _fields.size(); ++fi) {
                        _fields[fi].unpack(make_array(ii, jj, kk), _fields[fi].ptr, it);
                    }
                }
            }
        };
    };

} // namespace
#endif
