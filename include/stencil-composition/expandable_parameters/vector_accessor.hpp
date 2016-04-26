#pragma once

namespace gridtools{

    template < uint_t ID,
        enumtype::intend Intend = enumtype::in,
        typename Extent = extent< 0, 0, 0, 0, 0, 0 >,
        ushort_t NDim = 4 >
    struct vector_accessor : accessor<ID, Intend, Extent, NDim>{

        using super = accessor<ID, Intend, Extent, NDim>;
        using super::accessor;
        static const ushort_t n_dim = NDim;
    };

    template <typename T>
    struct is_vector_accessor : boost::mpl::false_ {};

    template <uint_t ID, enumtype::intend Intent, typename Extent, uint_t Size >
    struct is_vector_accessor<vector_accessor<ID, Intent, Extent, Size> > : boost::mpl::true_ {};

    // template <uint_t ID, enumtype::intend Intent, typename Extent, uint_t Size >
    // struct is_accessor<vector_accessor<ID, Intent, Extent, Size> > : boost::mpl::true_ {};


}//namespace gridtools
