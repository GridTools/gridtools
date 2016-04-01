#define MACRO_IMPL(z, n, _)                                                                                      \
    {                                                                                                            \
        const int ntx = 32;                                                                                      \
        const int nty = 1;                                                                                       \
        const int ntz = 8;                                                                                       \
        dim3 threads(ntx, nty, ntz);                                                                             \
                                                                                                                 \
        int nx = field##n.halos[0].s_length(-1) + field##n.halos[0].s_length(0) + field##n.halos[0].s_length(1); \
        int ny = field##n.halos[1].s_length(-1);                                                                 \
        int nz = field##n.halos[2].s_length(0);                                                                  \
                                                                                                                 \
        int nbx = (nx + ntx - 1) / ntx;                                                                          \
        int nby = (ny + nty - 1) / nty;                                                                          \
        int nbz = (nz + ntz - 1) / ntz;                                                                          \
        dim3 blocks(nbx, nby, nbz);                                                                              \
                                                                                                                 \
        if (nbx != 0 && nby != 0 && nbz != 0) {                                                                  \
            m_packYLKernel_generic<<< blocks, threads, 0, YL_stream >>>(field##n.ptr,                        \
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
