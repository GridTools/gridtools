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
#ifndef _DESCRIPTORS_DT_WHOLE_H_
#define _DESCRIPTORS_DT_WHOLE_H_

#include "../../common/array.hpp"
#include <vector>
#include "../../common/make_array.hpp"
#include <common/gt_assert.hpp>

#include "../../common/boollist.hpp"
#include "../../common/ndloops.hpp"
#include "../low-level/data_types_mapping.hpp"
#include "gcl_parameters.hpp"
#include "../../common/halo_descriptor.hpp"
#include "../../common/layout_map_metafunctions.hpp"

#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include "../../common/numerics.hpp"
#include "descriptors_fwd.hpp"
#include "descriptor_base.hpp"
#include "helpers_impl.hpp"
#include <boost/type_traits/remove_pointer.hpp>
#include <algorithm>

namespace gridtools {

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
        typename GT,
        typename proc_layout,
        typename Gcl_Arch,
        template < int Ndim > class GridType >
    class hndlr_dynamic_ut< DataType, GridType< 3 >, Halo_Exchange_3D_DT< GT >, proc_layout, Gcl_Arch, 1 >
        : public descriptor_base< Halo_Exchange_3D_DT< GT > > {
        static const int DIMS = 3;
        static const int MaxFields = 20;
        typedef descriptor_base< Halo_Exchange_3D_DT< GT > > base_type;
        typedef typename base_type::pattern_type HaloExch;
        typedef hndlr_dynamic_ut< DataType, GridType< 3 >, HaloExch, proc_layout, Gcl_Arch, 1 > this_type;
        typedef array< MPI_Datatype, _impl::static_pow3< DIMS >::value > MPDT_t;

        // typedef array<array<MPI_Datatype,MaxFields>, _impl::static_pow3<DIMS>::value> MPDT_array_t;
        // MPDT_array_t MPDT_array_in, MPDT_array_out;
      public:
        empty_field< DataType, DIMS > halo;

      private:
        gridtools::array< MPI_Aint, MaxFields > offsets; // 20 is the max number of fields passed in
        gridtools::array< int, MaxFields > counts;       // 20 is the max number of fields passed in

        MPDT_t MPDT_INSIDE, MPDT_OUTSIDE;

      public:
        typedef typename base_type::pattern_type pattern_type;

        /**
           Type of the computin grid associated to the pattern
         */
        typedef typename base_type::grid_type grid_type;

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
        template < typename Array >
        explicit hndlr_dynamic_ut(typename grid_type::period_type const &c, MPI_Comm comm, Array const *dimensions)
            : base_type(c, comm, dimensions), halo() {
            for (int i = 0; i < MaxFields; ++i)
                counts[i] = 1;
        }

        /**
           Constructor

           \param[in] c The object of the class used to specify periodicity in each dimension
           \param[in] _P Number of processors the pattern is running on (numbered from 0 to _P-1
           \param[in] _pid Integer identifier of the process calling the constructor
         */
        explicit hndlr_dynamic_ut(typename grid_type::period_type const &c, int _P, int _pid)
            : base_type(c, _P, _pid), halo() {
            for (int i = 0; i < MaxFields; ++i)
                counts[i] = 1;
        }

        /**
           Constructor

           \param[in] g A processor grid that will execute the pattern
         */
        explicit hndlr_dynamic_ut(grid_type const &g) : base_type(g), halo() {
            for (int i = 0; i < MaxFields; ++i)
                counts[i] = 1;
        }

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
        void setup(int) { halo.setup(); }

        /**
           Function to unpack received data

           \param[in] fields vector with data fields pointers to be packed from
        */
        void pack(std::vector< DataType * > const &fields) {
            // Create an MPI data types with data types of the different fields.
            for (unsigned int k = 0; k < fields.size(); ++k) {
                offsets[k] = reinterpret_cast< const char * >(fields[k]) - reinterpret_cast< const char * >(fields[0]);
            }

            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    for (int k = -1; k <= 1; ++k) {
                        if (i != 0 || j != 0 || k != 0) {
                            array< int, DIMS > eta = make_array(i, j, k);
                            if (halo.mpdt_inside(eta).second) {
                                MPI_Type_create_hindexed(fields.size(),
                                    &(counts[0]),
                                    &(offsets[0]),
                                    halo.mpdt_inside(eta).first,
                                    &(MPDT_INSIDE[_impl::neigh_idx(eta)]));
                                MPI_Type_commit(&MPDT_INSIDE[_impl::neigh_idx(eta)]);
                            }
                            if (halo.mpdt_outside(eta).second) {
                                MPI_Type_create_hindexed(fields.size(),
                                    &(counts[0]),
                                    &(offsets[0]),
                                    halo.mpdt_outside(eta).first,
                                    &(MPDT_OUTSIDE[_impl::neigh_idx(eta)]));
                                MPI_Type_commit(&MPDT_OUTSIDE[_impl::neigh_idx(eta)]);
                            }
                        }
                    }
                }
            }

            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    for (int k = -1; k <= 1; ++k) {
                        if (i != 0 || j != 0 || k != 0) {
                            array< int, DIMS > eta = make_array(i, j, k);
                            if (halo.mpdt_inside(eta).second) {
                                typedef translate_t< 3, proc_layout > translate_P;
                                typedef typename translate_P::map_type map_type;
                                const int i_P = pack_get_elem< map_type::template at< 0 >() >::apply(i, j, k);
                                const int j_P = pack_get_elem< map_type::template at< 1 >() >::apply(i, j, k);
                                const int k_P = pack_get_elem< map_type::template at< 2 >() >::apply(i, j, k);

                                base_type::m_haloexch.register_send_to_buffer(
                                    fields[0], MPDT_INSIDE[_impl::neigh_idx(eta)], 1, i_P, j_P, k_P);

                            } else {
                                typedef translate_t< 3, proc_layout > translate_P;
                                typedef typename translate_P::map_type map_type;
                                const int i_P = pack_get_elem< map_type::template at< 0 >() >::apply(i, j, k);
                                const int j_P = pack_get_elem< map_type::template at< 1 >() >::apply(i, j, k);
                                const int k_P = pack_get_elem< map_type::template at< 2 >() >::apply(i, j, k);

                                base_type::m_haloexch.register_send_to_buffer(NULL, MPI_INT, 0, i_P, j_P, k_P);
                            }

                            if (halo.mpdt_outside(eta).second) {
                                typedef translate_t< 3, proc_layout > translate_P;
                                typedef typename translate_P::map_type map_type;
                                const int i_P = pack_get_elem< map_type::template at< 0 >() >::apply(i, j, k);
                                const int j_P = pack_get_elem< map_type::template at< 1 >() >::apply(i, j, k);
                                const int k_P = pack_get_elem< map_type::template at< 2 >() >::apply(i, j, k);

                                base_type::m_haloexch.register_receive_from_buffer(
                                    fields[0], MPDT_OUTSIDE[_impl::neigh_idx(eta)], 1, i_P, j_P, k_P);
                            } else {
                                typedef translate_t< 3, proc_layout > translate_P;
                                typedef typename translate_P::map_type map_type;
                                const int i_P = pack_get_elem< map_type::template at< 0 >() >::apply(i, j, k);
                                const int j_P = pack_get_elem< map_type::template at< 1 >() >::apply(i, j, k);
                                const int k_P = pack_get_elem< map_type::template at< 2 >() >::apply(i, j, k);

                                base_type::m_haloexch.register_receive_from_buffer(NULL, MPI_INT, 0, i_P, j_P, k_P);
                            }
                        }
                    }
                }
            }
        }
        /**
           Function to unpack received data

           \param[in] fields vector with data fields pointers to be unpacked into
        */
        void unpack(std::vector< DataType * > const &fields) {
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    for (int k = -1; k <= 1; ++k) {
                        if (i != 0 || j != 0 || k != 0) {
                            array< int, DIMS > eta = make_array(i, j, k);
                            if (halo.mpdt_inside(eta).second) {
                                MPI_Type_free(&MPDT_INSIDE[_impl::neigh_idx(eta)]);
                            }
                            if (halo.mpdt_outside(eta).second) {
                                MPI_Type_free(&MPDT_OUTSIDE[_impl::neigh_idx(eta)]);
                            }
                        }
                    }
                }
            }
        }
    };

} // namespace

#endif
