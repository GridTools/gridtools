/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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
