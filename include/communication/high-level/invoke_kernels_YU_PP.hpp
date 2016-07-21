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
        const int nty = 1;                                                                                       \
        const int ntz = 8;                                                                                       \
        dim3 threads(ntx, nty, ntz);                                                                             \
                                                                                                                 \
        int nx = field##n.halos[0].s_length(-1) + field##n.halos[0].s_length(0) + field##n.halos[0].s_length(1); \
        int ny = field##n.halos[1].s_length(1);                                                                  \
        int nz = field##n.halos[2].s_length(0);                                                                  \
                                                                                                                 \
        int nbx = (nx + ntx - 1) / ntx;                                                                          \
        int nby = (ny + nty - 1) / nty;                                                                          \
        int nbz = (nz + ntz - 1) / ntz;                                                                          \
        dim3 blocks(nbx, nby, nbz);                                                                              \
                                                                                                                 \
        if (nbx != 0 && nby != 0 && nbz != 0) {                                                                  \
            m_packYUKernel_generic<<< blocks, threads, 0, YU_stream >>>(field##n.ptr,                        \
                reinterpret_cast< typename FOTF_T##n::value_type ** >(d_msgbufTab),                              \
                wrap_argument(d_msgsize + 27 * n),                                                               \
                *(reinterpret_cast< const gridtools::array< gridtools::halo_descriptor, 3 > * >(&field##n)),     \
                nx,                                                                                              \
                nz,                                                                                              \
                0);                                                                                              \
        }                                                                                                        \
    }

BOOST_PP_REPEAT(noi, MACRO_IMPL, all)
#undef MACRO_IMPL
