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
#ifndef _DESCRIPTORS_H_
#define _DESCRIPTORS_H_

#include "../../common/array.hpp"
#include <vector>
#include "../low-level/proc_grids_3D.hpp"
#include "../low-level/Halo_Exchange_3D.hpp"
#include "../../common/make_array.hpp"
#include "../../common/generic_metafunctions/pack_get_elem.hpp"
#include <common/gt_assert.hpp>
#include <boost/type_traits/remove_pointer.hpp>

#include "../../common/boollist.hpp"
#include "gcl_parameters.hpp"
#include "empty_field_base.hpp"
#include "../low-level/translate.hpp"
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include "../../common/numerics.hpp"
#include "descriptors_fwd.hpp"
#include "descriptor_base.hpp"
#include "helpers_impl.hpp"
#include "access.hpp"

namespace gridtools {

    /** \class empty_field_no_dt
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
    template < int DIMS >
    class empty_field_no_dt : public empty_field_base< int, DIMS > {

        typedef empty_field_base< int, DIMS > base_type;

      public:
        /**
            Constructor that receive the pointer to the data. This is explicit and
            must then be called.
        */
        explicit empty_field_no_dt() {}

        void setup() const {}

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
        void pack(gridtools::array< int, 2 > const &eta, iterator_in const *field_ptr, iterator_out *&it) const {
            for (int j = base_type::halos[1].loop_low_bound_inside(eta[1]);
                 j <= base_type::halos[1].loop_high_bound_inside(eta[1]);
                 ++j) {
                for (int i = base_type::halos[0].loop_low_bound_inside(eta[0]);
                     i <= base_type::halos[0].loop_high_bound_inside(eta[0]);
                     ++i) {
                    *(reinterpret_cast< iterator_in * >(it)) = field_ptr[gridtools::access(
                        i, j, base_type::halos[0].total_length(), base_type::halos[1].total_length())];
                    reinterpret_cast< char *& >(it) += sizeof(iterator_in);
                }
            }
        }

        template < typename iterator_in, typename iterator_out >
        void pack(gridtools::array< int, 3 > const &eta, iterator_in const *field_ptr, iterator_out *&it) const {
            // std::cout << "BASE ADDR IN PACK " << std::hex << reinterpret_cast<void*>(it) << std::dec << std::endl;
            // std::cout << "k=" << base_type::halos[2].loop_low_bound_inside(eta[2])
            //           << " to <= " << base_type::halos[2].loop_high_bound_inside(eta[2]) << ",\n"
            //           << "j=" << base_type::halos[1].loop_low_bound_inside(eta[1])
            //           << " to <= " << base_type::halos[1].loop_high_bound_inside(eta[1]) << ",\n"
            //           << "i=" << base_type::halos[0].loop_low_bound_inside(eta[0])
            //           << " to <= " << base_type::halos[0].loop_high_bound_inside(eta[0])
            //           << std::endl;
            for (int k = base_type::halos[2].loop_low_bound_inside(eta[2]);
                 k <= base_type::halos[2].loop_high_bound_inside(eta[2]);
                 ++k) {
                for (int j = base_type::halos[1].loop_low_bound_inside(eta[1]);
                     j <= base_type::halos[1].loop_high_bound_inside(eta[1]);
                     ++j) {
                    for (int i = base_type::halos[0].loop_low_bound_inside(eta[0]);
                         i <= base_type::halos[0].loop_high_bound_inside(eta[0]);
                         ++i) {

                        // std::cout << "      "
                        //           << i << ", "
                        //           << j << ", "
                        //           << k << ", "
                        //           << base_type::halos[0].total_length() << ", "
                        //           << base_type::halos[1].total_length() << ", "
                        //           << base_type::halos[2].total_length() << ", "
                        //           << field_ptr[gridtools::access
                        //                        (i,j,k,
                        //                         base_type::halos[0].total_length(),
                        //                         base_type::halos[1].total_length(),
                        //                         base_type::halos[2].total_length())]
                        //           << std::endl;

                        *(reinterpret_cast< iterator_in * >(it)) = field_ptr[gridtools::access(i,
                            j,
                            k,
                            base_type::halos[0].total_length(),
                            base_type::halos[1].total_length(),
                            base_type::halos[2].total_length())];
                        reinterpret_cast< char *& >(it) += sizeof(iterator_in);
                    }
                }
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
        void unpack(gridtools::array< int, 2 > const &eta, iterator_in *field_ptr, iterator_out *&it) const {
            for (int j = base_type::halos[1].loop_low_bound_outside(eta[1]);
                 j <= base_type::halos[1].loop_high_bound_outside(eta[1]);
                 ++j) {
                for (int i = base_type::halos[0].loop_low_bound_outside(eta[0]);
                     i <= base_type::halos[0].loop_high_bound_outside(eta[0]);
                     ++i) {
                    field_ptr[gridtools::access(
                        i, j, base_type::halos[0].total_length(), base_type::halos[1].total_length())] =
                        *(reinterpret_cast< iterator_in * >(it));
                    reinterpret_cast< char *& >(it) += sizeof(iterator_in);
                }
            }
        }

        template < typename iterator_in, typename iterator_out >
        void unpack(gridtools::array< int, 3 > const &eta, iterator_in *field_ptr, iterator_out *&it) const {
            // std::cout << "k=" << base_type::halos[2].loop_low_bound_outside(eta[2])
            //           << " to <= " << base_type::halos[2].loop_high_bound_outside(eta[2]) << ",\n"
            //           << "j=" << base_type::halos[1].loop_low_bound_outside(eta[1])
            //           << " to <= " << base_type::halos[1].loop_high_bound_outside(eta[1]) << ",\n"
            //           << "i=" << base_type::halos[0].loop_low_bound_outside(eta[0])
            //           << " to <= " << base_type::halos[0].loop_high_bound_outside(eta[0])
            //           << std::endl;
            for (int k = base_type::halos[2].loop_low_bound_outside(eta[2]);
                 k <= base_type::halos[2].loop_high_bound_outside(eta[2]);
                 ++k) {
                for (int j = base_type::halos[1].loop_low_bound_outside(eta[1]);
                     j <= base_type::halos[1].loop_high_bound_outside(eta[1]);
                     ++j) {
                    for (int i = base_type::halos[0].loop_low_bound_outside(eta[0]);
                         i <= base_type::halos[0].loop_high_bound_outside(eta[0]);
                         ++i) {
                        // std::cout << "      "
                        //           << i << ", "
                        //           << j << ", "
                        //           << k << ", "
                        //           << base_type::halos[0].total_length() << ", "
                        //           << base_type::halos[1].total_length() << ", "
                        //           << base_type::halos[2].total_length() << ", "
                        //           << *(reinterpret_cast<iterator_in*>(it))
                        //           << std::endl;
                        field_ptr[gridtools::access(i,
                            j,
                            k,
                            base_type::halos[0].total_length(),
                            base_type::halos[1].total_length(),
                            base_type::halos[2].total_length())] = *(reinterpret_cast< iterator_in * >(it));
                        reinterpret_cast< char *& >(it) += sizeof(iterator_in);
                    }
                }
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
        template < typename iterator, typename FIRST, typename... FIELDS >
        void pack_all(
            gridtools::array< int, DIMS > const &eta, iterator &it, FIRST const &field, const FIELDS &... args) const {
            pack(eta, field, it);
            pack_all(eta, it, args...);
        }

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
           \param[in] args the rest of the list of data fields where data has to be unpacked into (they may have
           different
           datatypes).
        */
        template < typename iterator, typename FIRST, typename... FIELDS >
        void unpack_all(
            gridtools::array< int, DIMS > const &eta, iterator &it, FIRST const &field, const FIELDS &... args) const {
            unpack(eta, field, it);
            unpack_all(eta, it, args...);
        }
    };

    template < int I >
    std::ostream &operator<<(std::ostream &s, empty_field_no_dt< I > const &ef) {
        s << "empty_field_no_dt ";
        for (int i = 0; i < I; ++i)
            s << ef.raw_array()[i] << ", ";
        return s;
    }

    /** \class field_descriptor_no_dt
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
    class field_descriptor_no_dt : public empty_field_no_dt< DIMS > {
        DataType *fieldptr; // Pointer to the data field

        typedef empty_field_no_dt< DIMS > base_type;

      public:
        /**
            Constructor that receive the pointer to the data. This is explicit and
            must then be called.
            \param[in] _fp DataType* pointer to the data field
        */
        explicit field_descriptor_no_dt(DataType *_fp) : fieldptr(_fp) {}

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
        Class containing the list of data fields associated with an handler. A handler
        identifies the data fileds that must be updated together in the computation.

        The _ut suffix stand for "uniform type", that is, all the data fields in this
        descriptor have the same data type, which is equal to the template argument.

        The order in which data fields are registered is important, since it dictated the order
        in which the data is packed, transfered and unpacked. All processes must register
        the data fields in the order and with the same corresponding sizes.

        \tparam DataType type of the elements of the data fields associated to the handler.
        \tparam DIMS Number of dimensions of the grids.
        \tparam HaloExch Communication patter with halo exchange.
    */
    template < typename DataType, int DIMS, typename HaloExch >
    class hndlr_descriptor_ut : public descriptor_base< HaloExch > {
        typedef hndlr_descriptor_ut< DataType, DIMS, HaloExch > this_type;

        std::vector< field_descriptor_no_dt< DataType, DIMS > > field;

        gridtools::array< DataType *, _impl::static_pow3< DIMS >::value > send_buffer; // One entry will not be used...
        gridtools::array< DataType *, _impl::static_pow3< DIMS >::value > recv_buffer;

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

      private:
        hndlr_descriptor_ut(hndlr_descriptor_ut const &) {}

      public:
        /**
           Constructor

           \param[in] c The object of the class used to specify periodicity in each dimension
           \param[in] comm MPI communicator (typically MPI_Comm_world)
        */
        explicit hndlr_descriptor_ut(typename grid_type::period_type const &c, MPI_Comm comm)
            : base_type(grid_type(c, comm)), field() {}

        /**
           Constructor

           \param[in] c The object of the class used to specify periodicity in each dimension
           \param[in] _P Number of processors the pattern is running on (numbered from 0 to _P-1
           \param[in] _pid Integer identifier of the process calling the constructor
        */
        explicit hndlr_descriptor_ut(typename grid_type::period_type const &c, int _P, int _pid)
            : base_type(grid_type(c, _P, _pid)), field() {}

        /**
           Constructor

           \param[in] g A processor grid that will execute the pattern
         */
        explicit hndlr_descriptor_ut(grid_type const &g) : base_type(g), field() {}

        /**
           Add a data field to the handler descriptor. Returns the index of the field
           for later use.

           \param[in] ptr pointer to the datafield
           \return index of the field in the handler desctiptor
        */
        size_t register_field(DataType *ptr) {
            field.push_back(field_descriptor_no_dt< DataType, DIMS >(ptr));
            return field.size() - 1;
        }

        /**
           Register the halo relative to a given dimension with a given data field/

           \param[in] D index of data field to be affected
           \param[in] I index of dimension for which the information is passed
           \param[in] minus Please see field_descriptor_no_dt, halo_descriptor or \link MULTI_DIM_ACCESS \endlink for
           details
           \param[in] plus Please see field_descriptor_no_dt, halo_descriptor or \link MULTI_DIM_ACCESS \endlink for
           details
           \param[in] begin Please see field_descriptor_no_dt, halo_descriptor or \link MULTI_DIM_ACCESS \endlink for
           details
           \param[in] end Please see field_descriptor_no_dt, halo_descriptor or \link MULTI_DIM_ACCESS \endlink for
           details
           \param[in] t_len Please see field_descriptor_no_dt, halo_descriptor or \link MULTI_DIM_ACCESS \endlink for
           details
        */
        void register_halo(size_t D, size_t I, int minus, int plus, int begin, int end, int t_len) {
            field[D].add_halo(I, minus, plus, begin, end, t_len);
        }

        int size() const { return field.size(); }

        field_descriptor_no_dt< DataType, DIMS > const &data_field(int I) const { return field[I]; }

        /** Given the coordinates of a neighbor (2D), return the total number of elements
            to be sent to that neighbor associated with the handler of the manager.
        */
        template < typename ARRAY >
        int total_pack_size(ARRAY const &tuple) const {
            int S = 0;
            for (int i = 0; i < size(); ++i)
                S += data_field(i).send_buffer_size(tuple);
            return S;
        }

        /** Given the coordinates of a neighbor (2D), return the total number of elements
            to be received from that neighbor associated with the handler of the manager.
        */
        template < typename ARRAY >
        int total_unpack_size(ARRAY const &tuple) const {
            int S = 0;
            for (int i = 0; i < size(); ++i)
                S += data_field(i).recv_buffer_size(tuple);
            return S;
        }

        /**
           Function to setup internal data structures for data exchange and preparing eventual underlying layers

           The use of this function is deprecated
        */
        void allocate_buffers() { setup(); }

        /**
           Function to setup internal data structures for data exchange and preparing eventual underlying layers
        */
        void setup() { _impl::allocation_service< this_type >()(this); }

        /**
           Function to pack data to be sent
        */
        void pack() const { _impl::pack_service< this_type >()(this); }

        /**
           Function to unpack received data
        */
        void unpack() const { _impl::unpack_service< this_type >()(this); }

        /// Utilities

        /**
           Retrieve the pattern from which the computing grid and other information
           can be retrieved. The function is available only if the underlying
           communication library is a Level 3 pattern. It would not make much
           sense otherwise.

           If used to get process grid information additional information can be
           found in \link GRIDS_INTERACTION \endlink
        */
        pattern_type const &pattern() const { return base_type::m_haloexch; }

        // FRIENDING
        friend class _impl::allocation_service< this_type >;
        friend class _impl::pack_service< this_type >;
        friend class _impl::unpack_service< this_type >;
    };

    /**
        Class containing the description of one halo and a communication
        pattern.  A communication is triggered when a list of data
        fields are passed to the exchange functions, when the data
        according to the halo descriptors are exchanged. This class is
        needed when the addresses and the number of the data fields
        changes dynamically but the sizes are constant. Data elements
        for each hndlr_dynamic_ut must be the same.

        \tparam DIMS Number of dimensions of the grids.
        \tparam HaloExch Communication pattern with halo exchange.
    */
    template < typename DataType, typename HaloExch, typename proc_layout, class GridType >
    class hndlr_dynamic_ut< DataType, GridType, HaloExch, proc_layout, gcl_cpu, 2 >
        : public descriptor_base< HaloExch > {

        static const int DIMS = GridType::ndims;
        typedef hndlr_dynamic_ut< DataType, GridType, HaloExch, proc_layout, gcl_cpu, 2 > this_type;

      public:
        empty_field_no_dt< DIMS > halo;

      private:
        gridtools::array< DataType *, _impl::static_pow3< DIMS >::value > send_buffer; // One entry will not be used...
        gridtools::array< DataType *, _impl::static_pow3< DIMS >::value > recv_buffer;

      public:
        typedef gcl_cpu arch_type;
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

      private:
        hndlr_dynamic_ut(hndlr_dynamic_ut const &) {}

      public:
        /**
           Constructor

           \param[in] c The object of the class used to specify periodicity in each dimension
           \param[in] comm MPI communicator (typically MPI_Comm_world)
        */
        explicit hndlr_dynamic_ut(typename grid_type::period_type const &c, MPI_Comm const &comm)
            : base_type(c, comm), halo() {}

        ~hndlr_dynamic_ut() {
#ifdef _GCL_CHECK_DESTRUCTOR
            std::cout << "Destructor " << __FILE__ << ":" << __LINE__ << std::endl;
#endif

            _destroy_dynamic_ut< DIMS, 0 >().do_it(this);
        }

        /**
           Constructor

           \param[in] c The object of the class used to specify periodicity in each dimension
           \param[in] _P Number of processors the pattern is running on (numbered from 0 to _P-1
           \param[in] _pid Integer identifier of the process calling the constructor
         */
        explicit hndlr_dynamic_ut(typename grid_type::period_type const &c, int _P, int _pid)
            : halo(), base_type::m_haloexch(grid_type(c, _P, _pid)) // procgrid)
        {}

        /**
           Constructor

           \param[in] g A processor grid that will execute the pattern
         */
        explicit hndlr_dynamic_ut(grid_type const &g) : halo(), base_type::m_haloexch(g) {}

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
        void setup(int max_fields_n) { _impl::allocation_service< this_type >()(this, max_fields_n); }

#ifdef GCL_TRACE
        void set_pattern_tag(int tag) { base_type::m_haloexch.set_pattern_tag(tag); };
#endif

        /**
           Function to pack data to be sent

           \param[in] _fields data fields to be packed
        */
        template < typename... FIELDS >
        void pack(const FIELDS &... _fields) const {
            pack_dims< DIMS, 0 >()(*this, _fields...);
        }

        /**
           Function to unpack received data

           \param[in] _fields data fields where to unpack data
        */
        template < typename... FIELDS >
        void unpack(const FIELDS &... _fields) const {
            unpack_dims< DIMS, 0 >()(*this, _fields...);
        }

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

        /**
           Retrieve the pattern from which the computing grid and other information
           can be retrieved. The function is available only if the underlying
           communication library is a Level 3 pattern. It would not make much
           sense otherwise.

           If used to get process grid information additional information can be
           found in \link GRIDS_INTERACTION \endlink
        */
        pattern_type const &pattern() const { return base_type::pattern(); }

        // FRIENDING
        friend class _impl::allocation_service< this_type >;
        // friend class _impl::pack_service<this_type>;
        // friend class _impl::unpack_service<this_type>;

      private:
        template < int I, int dummy >
        struct pack_dims {};

        template < int dummy >
        struct pack_dims< 3, dummy > {
            template < typename T, typename... FIELDS >
            void operator()(const T &hm, const FIELDS &... _fields) const {
#pragma omp parallel for schedule(dynamic, 1) collapse(3)
                for (int ii = -1; ii <= 1; ++ii) {
                    for (int jj = -1; jj <= 1; ++jj) {
                        for (int kk = -1; kk <= 1; ++kk) {
                            typedef proc_layout map_type;
                            const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                            const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                            const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
                            if ((ii != 0 || jj != 0 || kk != 0) &&
                                (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                                DataType *it = &(hm.send_buffer[translate()(ii, jj, kk)][0]);
                                hm.halo.pack_all(make_array(ii, jj, kk), it, _fields...);
                            }
                        }
                    }
                }
            }
        };

        template < int I, int dummy >
        struct unpack_dims {};

        template < int dummy >
        struct unpack_dims< 3, dummy > {
            template < typename T, typename... FIELDS >
            void operator()(const T &hm, const FIELDS &... _fields) const {
#pragma omp parallel for schedule(dynamic, 1) collapse(3)
                for (int ii = -1; ii <= 1; ++ii) {
                    for (int jj = -1; jj <= 1; ++jj) {
                        for (int kk = -1; kk <= 1; ++kk) {
                            typedef proc_layout map_type;
                            const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                            const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                            const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
                            if ((ii != 0 || jj != 0 || kk != 0) &&
                                (hm.pattern().proc_grid().proc(ii_P, jj_P, kk_P) != -1)) {
                                DataType *it = &(hm.recv_buffer[translate()(ii, jj, kk)][0]);
                                hm.halo.unpack_all(make_array(ii, jj, kk), it, _fields...);
                            }
                        }
                    }
                }
            }
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
                            const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                            const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                            const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
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
                            const int ii_P = pack_get_elem< map_type::template at< 0 >() >::apply(ii, jj, kk);
                            const int jj_P = pack_get_elem< map_type::template at< 1 >() >::apply(ii, jj, kk);
                            const int kk_P = pack_get_elem< map_type::template at< 2 >() >::apply(ii, jj, kk);
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

        template < int D, int Dummy >
        struct _destroy_dynamic_ut {};

        template < int Dummy >
        struct _destroy_dynamic_ut< 3, Dummy > {
            template < typename T >
            void do_it(T hm) const {
                for (int i = -1; i <= 1; ++i) {
                    for (int j = -1; j <= 1; ++j) {
                        for (int k = -1; k <= 1; ++k) {
                            if (!hm->send_buffer[translate()(i, j, k)])
                                _impl::gcl_alloc< DataType, arch_type >::free(hm->send_buffer[translate()(i, j, k)]);
                            if (!hm->recv_buffer[translate()(i, j, k)])
                                _impl::gcl_alloc< DataType, arch_type >::free(hm->recv_buffer[translate()(i, j, k)]);
                        }
                    }
                }
            }
        };
    };

} // namespace

#endif
