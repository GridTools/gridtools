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
#ifndef DATA_TYPES_MAPPING_H_
#define DATA_TYPES_MAPPING_H_

namespace gridtools {
    namespace _impl {
        template < typename v_type >
        struct compute_type {
            // This is called when no other type is found.
            // In this case the type is considered POD
            // No static value available here
            MPI_Datatype value;

            compute_type() {
                MPI_Type_contiguous(sizeof(v_type), MPI_CHAR, &value);
                MPI_Type_commit(&value);
            }
        };

        template <>
        struct compute_type< int > {
            const MPI_Datatype value;
            compute_type() : value(MPI_INT) {}
        };

        template <>
        struct compute_type< char > {
            const MPI_Datatype value;
            compute_type() : value(MPI_CHAR) {}
        };

        template <>
        struct compute_type< float > {
            const MPI_Datatype value;
            compute_type() : value(MPI_FLOAT) {}
        };

        template <>
        struct compute_type< double > {
            const MPI_Datatype value;
            compute_type() : value(MPI_DOUBLE) {}
        };

        template < typename value_type >
        struct make_datatype {

          public:
            static MPI_Datatype type() { return _impl::compute_type< value_type >().value; }

            template < typename arraytype >
            static MPI_Datatype make(arraytype const &halo) {
                const int d = halo.size();
                std::vector< int > sizes(d), subsizes(d), starts(d);

                for (int i = 0; i < d; ++i) {
                    sizes[i] = halo[i].total_length();
                    subsizes[i] = halo[i].end() - halo[i].begin() + 1;
                    starts[i] = halo[i].begin();
                }

                MPI_Datatype res;
                MPI_Type_create_subarray(d,
                    &sizes[0],
                    &subsizes[0],
                    &starts[0],
                    MPI_ORDER_FORTRAN, // increasing strides
                    type(),
                    &res);
                MPI_Type_commit(&res);
                return res;
            }
        };
    } // namespace _impl
}
#endif
