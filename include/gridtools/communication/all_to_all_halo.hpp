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
#ifndef _ALL_TO_ALL_HALO_H_
#define _ALL_TO_ALL_HALO_H_

#include "low-level/access_functions.hpp"
#include "../common/halo_descriptor.hpp"
#include "low-level/Generic_All_to_All.hpp"
#include "low-level/data_types_mapping.hpp"

/**
@file
*/

namespace gridtools {

    template < typename vtype, typename pgrid >
    struct all_to_all_halo {
        /** Type of the elements to be exchanged.
         */
        typedef vtype value_type;

        /** Type of the processing grid used for data exchange
         */
        typedef pgrid grid_type;

        /** Number of dimensions of the computing grid
         */
        static const int ndims = grid_type::ndims;

      private:
        const grid_type proc_grid;
        all_to_all< value_type > a2a;

      public:
        /** Constructor that takes the computing grid and initializes the
            patern.
         */
        all_to_all_halo(grid_type const &g) : proc_grid(g), a2a(proc_grid.size()) {}

        /** Constructor that takes the computing grid and initializes the
            pattern. It also takes a communicator that is inside the processor
            grid, if different from MPI_COMM_WORLD
         */
        all_to_all_halo(grid_type const &g, MPI_Comm &c) : proc_grid(g), a2a(proc_grid.size(), c) {}

        /** This function takes an array or vector of halos (sorted by
            decreasing strides) (size equal to ndims), the pointer to the
            data and the coordinated of the receiving processors and
            prepare the pattern to send that sub-array to that processor.

            \tparam arraytype1 type of the array of halos. This is required to have only the operator[] and the method
           size()
            \tparam arraytype2 type of the array of coordinates of the destination process. This is required to have
           only the operator[]

            \param field Pointer to the data do be sent
            \param Array or vector of type arraytype1 that contains the description of the data to be sent
            \param coords Array of vector of absolute coordinates of the process that will receive the data
         */
        template < typename arraytype1, typename arraytype2 >
        void register_block_to(value_type *field, arraytype1 const &halo_block, arraytype2 const &coords) {
#ifndef NDEBUG
            std::cout << "register_block_to " << proc_grid.abs_proc(coords) << "\n";
#endif
            a2a.to[proc_grid.abs_proc(coords)] =
                packet< value_type >(_impl::make_datatype< value_type >::make(halo_block), field);
        }

        /** This function takes an array or vector of halos (sorted by
            decreasing strides) (size equal to ndims), the pointer to the
            data and the coordinated of the receiving processors and
            prepare the pattern to receive that sub-array from that
            processor.

            \tparam arraytype1 type of the array of halos. This is required to have only the operator[] and the method
           size()
            \tparam arraytype2 type of the array of coordinates of the destination process. This is required to have
           only the operator[]

            \param field Pointer to the data do be received
            \param Array or vector of type arraytype1 that contains the description of the data to be received
            \param coords Array of vector of absolute coordinates of the process that from where the data will be
           received
         */
        template < typename arraytype1, typename arraytype2 >
        void register_block_from(value_type *field, arraytype1 const &halo_block, arraytype2 const &coords) {
#ifndef NDEBUG
            std::cout << "register_block_from " << proc_grid.abs_proc(coords) << "\n";
#endif
            a2a.from[proc_grid.abs_proc(coords)] =
                packet< value_type >(_impl::make_datatype< value_type >::make(halo_block), field);
        }

        /** This method prepare the pattern to be ready to execute
         */
        void setup() { a2a.setup(); }

        /**
         * Method to post receives
         */
        void post_receives() { a2a.post_receives(); }

        /**
         * Method to send data
         */
        void do_sends() { a2a.do_sends(); }

        /** This method starts the data exchange
         */
        void start_exchange() { a2a.start_exchange(); }

        /** This method waits for the data to arrive and be unpacked
         */
        void wait() { a2a.wait(); }
    };
} // namespace gridtools
#endif
