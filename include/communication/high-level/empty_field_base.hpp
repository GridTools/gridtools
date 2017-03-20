/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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
#ifndef _EMPTY_FIELD_BASE_H_
#define _EMPTY_FIELD_BASE_H_

#include "../../common/boollist.hpp"
#include "../../common/ndloops.hpp"
#include "../low-level/data_types_mapping.hpp"
#include "../../common/numerics.hpp"
#include "halo_descriptor.hpp"

namespace gridtools {
    namespace _impl {
        template < typename value_type >
        struct make_datatype_outin {

          public:
            static MPI_Datatype type() { return compute_type< value_type >().value; }

            template < typename arraytype, typename arraytype2 >
            static std::pair< MPI_Datatype, bool > outside(arraytype const &halo, arraytype2 const &eta) {
                const int d = halo.size();
                std::vector< int > sizes(d), subsizes(d), starts(d);

                int ssz = 1;
                for (int i = 0; i < d; ++i) {
                    sizes[i] = halo[i].total_length();
                    subsizes[i] = halo[i].loop_high_bound_outside(eta[i]) - halo[i].loop_low_bound_outside(eta[i]) + 1;
                    ssz *= subsizes[i];
                    starts[i] = halo[i].loop_low_bound_outside(eta[i]);
                }

                if (ssz == 0)
                    return std::make_pair(MPI_INT, false);

                MPI_Datatype res;
                MPI_Type_create_subarray(d,
                    &sizes[0],
                    &subsizes[0],
                    &starts[0],
                    MPI_ORDER_FORTRAN, // increasing strides
                    type(),
                    &res);
                MPI_Type_commit(&res);
                return std::make_pair(res, true);
            }

            template < typename arraytype, typename arraytype2 >
            static std::pair< MPI_Datatype, bool > inside(arraytype const &halo, arraytype2 const &eta) {
                const int d = halo.size();
                std::vector< int > sizes(d), subsizes(d), starts(d);

                int ssz = 1;
                for (int i = 0; i < d; ++i) {
                    sizes[i] = halo[i].total_length();
                    subsizes[i] = halo[i].loop_high_bound_inside(eta[i]) - halo[i].loop_low_bound_inside(eta[i]) + 1;
                    ssz *= subsizes[i];
                    starts[i] = halo[i].loop_low_bound_inside(eta[i]);
                }

                if (ssz == 0)
                    return std::make_pair(MPI_INT, false);

                MPI_Datatype res;
                MPI_Type_create_subarray(d,
                    &sizes[0],
                    &subsizes[0],
                    &starts[0],
                    MPI_ORDER_FORTRAN, // increasing strides
                    type(),
                    &res);
                MPI_Type_commit(&res);
                return std::make_pair(res, true);
            }
        };

        template < typename Array >
        int neigh_idx(Array const &tuple) {
            int idx = 0;
            for (std::size_t i = 0; i < tuple.size(); ++i) {
                int prod = 1;
                for (std::size_t j = 0; j < i; ++j) {
                    prod = prod * 3;
                }
                idx = idx + (tuple[i] + 1) * prod;
            }
            return idx;
        }

        template < typename DataType, typename HALO_t, typename MPDT_t >
        struct set_mpdt {
            MPDT_t &mpdt_out;
            MPDT_t &mpdt_in;
            HALO_t const &halo;
            set_mpdt(HALO_t const &h, MPDT_t &t, MPDT_t &e) : mpdt_out(t), mpdt_in(e), halo(h) {}

            template < typename Tuple >
            void operator()(Tuple const &tuple) {
                int idx = neigh_idx(tuple);
                mpdt_out[idx] = _impl::make_datatype_outin< DataType >::outside(halo, tuple);
                mpdt_in[idx] = _impl::make_datatype_outin< DataType >::inside(halo, tuple);
            }
        };
    }

    template < typename DataType, int DIMS >
    class empty_field_base {
        typedef array< halo_descriptor, DIMS > HALO_t;

      public:
        array< halo_descriptor, DIMS > halos;
        typedef array< std::pair< MPI_Datatype, bool >, _impl::static_pow3< DIMS >::value > MPDT_t;
        MPDT_t MPDT_OUTSIDE;
        MPDT_t MPDT_INSIDE;

      private:
        void generate_datatypes() {
            array< int, DIMS > tuple;
            _impl::set_mpdt< DataType, HALO_t, MPDT_t > set_function(halos, MPDT_OUTSIDE, MPDT_INSIDE);
            neigh_loop< DIMS > nl;
            nl(set_function, tuple);
        }

      public:
        /**
            Function to set the halo descriptor of the field descriptor

            \param[in] D index of the dimension to be set
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
        void add_halo(int D, int minus, int plus, int begin, int end, int t_len) {
            halos[D] = halo_descriptor(minus, plus, begin, end, t_len);
        }

        void add_halo(int D, halo_descriptor const &halo) { halos[D] = halo; }

        void setup() { generate_datatypes(); }

        std::pair< MPI_Datatype, bool > mpdt_inside(gridtools::array< int, DIMS > const &eta) const {
            return MPDT_INSIDE[_impl::neigh_idx(eta)];
        }

        std::pair< MPI_Datatype, bool > mpdt_outside(gridtools::array< int, DIMS > const &eta) const {
            return MPDT_OUTSIDE[_impl::neigh_idx(eta)];
        }

        /**
            Return the number of elements (not bytes) that have to be sent to a the neighbor
            indicated as an argument. This is the product of the lengths as in
            \link MULTI_DIM_ACCESS \endlink

            \param[in] eta the eta parameter as indicated in \link MULTI_DIM_ACCESS \endlink
        */
        int send_buffer_size(gridtools::array< int, DIMS > const &eta) const {
            int S = 1;
            for (int i = 0; i < DIMS; ++i) {
                S *= halos[i].s_length(eta[i]);
            }
            return S;
        }

        /**
            Return the number of elements (not bytes) that be receiver from the the neighbor
            indicated as an argument. This is the product of the lengths as in
            \link MULTI_DIM_ACCESS \endlink

            \param[in] eta the eta parameter as indicated in \link MULTI_DIM_ACCESS \endlink
        */
        int recv_buffer_size(gridtools::array< int, DIMS > const &eta) const {
            int S = 1;
            for (int i = 0; i < DIMS; ++i) {
                S *= halos[i].r_length(eta[i]);
            }
            return S;
        }
    };
}
#endif
