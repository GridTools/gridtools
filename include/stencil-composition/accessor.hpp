#pragma once

//TODOMEETING protect this with define and ifdef inside
//TODOMEETING inline namespaces to protect grid backends

namespace gridtools {
    template <uint_t I, enumtype::intend Intend>
    struct global_accessor{

        typedef global_accessor<I, Intend> type;
        // static const ushort_t n_dim=Dim;
        typedef static_uint<I> index_type;
        // typedef Range range_type;
    };

}//namespace gridtools

#ifdef STRUCTURED_GRIDS
    #include "stencil-composition/structured_grids/accessor_metafunctions.hpp"
    #include "stencil-composition/structured_grids/accessor.hpp"
#else
    #include "stencil-composition/icosahedral_grids/accessor_metafunctions.hpp"
    #include "stencil-composition/icosahedral_grids/accessor.hpp"
#endif
