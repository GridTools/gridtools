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
#define MACRO_IMPL(z, n, _)                                                                                      \
    {                                                                                                            \
        const int ntx = 32;                                                                                      \
        const int nty = 8;                                                                                       \
        const int ntz = 1;                                                                                       \
        dim3 threads(ntx, nty, ntz);                                                                             \
                                                                                                                 \
        int nx = field##n.halos[0].r_length(-1) + field##n.halos[0].r_length(0) + field##n.halos[0].r_length(1); \
        int ny = field##n.halos[1].r_length(-1) + field##n.halos[1].r_length(0) + field##n.halos[1].r_length(1); \
        int nz = field##n.halos[2].r_length(-1);                                                                 \
                                                                                                                 \
        int nbx = (nx + ntx - 1) / ntx;                                                                          \
        int nby = (ny + nty - 1) / nty;                                                                          \
        int nbz = (nz + ntz - 1) / ntz;                                                                          \
        dim3 blocks(nbx, nby, nbz);                                                                              \
                                                                                                                 \
        if (nbx != 0 && nby != 0 && nbz != 0) {                                                                  \
            m_unpackZLKernel_generic<<< blocks, threads, 0, ZL_stream >>>(field##n.ptr,                      \
                reinterpret_cast< typename FOTF_T##n::value_type ** >(d_msgbufTab_r),                            \
                wrap_argument(d_msgsize_r + 27 * n),                                                             \
                *(reinterpret_cast< const gridtools::array< gridtools::halo_descriptor, 3 > * >(&field##n)),     \
                nx,                                                                                              \
                ny,                                                                                              \
                (field##n.halos[0].begin() - field##n.halos[0].minus()) +                                        \
                    (field##n.halos[1].begin() - field##n.halos[1].minus()) * field##n.halos[0].total_length() + \
                    (field##n.halos[2].begin() - field##n.halos[2].minus()) * field##n.halos[0].total_length() * \
                        field##n.halos[1].total_length(),                                                        \
                0);                                                                                              \
        }                                                                                                        \
    }

BOOST_PP_REPEAT(noi, MACRO_IMPL, all)
#undef MACRO_IMPL
