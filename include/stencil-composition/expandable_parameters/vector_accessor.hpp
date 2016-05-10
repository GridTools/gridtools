#pragma once
/** @file vector accessor */

namespace gridtools{

/**
   @brief accessor for an expandable parameters list

   accessor object used with the expandable parameters. It is exactly like a regular accessor.
   Its type must be different though, so that the gridtools::iterate_domain can implement a specific
   overload of the operator() for this accessor type.

   \tparam ID integer identifier, to univocally specify the accessor
   \tparam Intent flag stating wether or not this accessor is read only
   \tparam Extent specification of the minimum box containing the stencil access pattern
   \tparam NDim dimensionality of the vector accessor: should be the storage space dimensions plus one (the vector field dimension)
*/
    template < uint_t ID,
        enumtype::intend Intent = enumtype::in,
        typename Extent = extent< 0, 0, 0, 0, 0, 0 >,
        ushort_t NDim = 4 >
    struct vector_accessor : accessor<ID, Intent, Extent, NDim>{

#ifdef CXX11_ENABLED
        using super = accessor<ID, Intent, Extent, NDim>;
        using super::accessor;
        static const ushort_t n_dim = NDim;
#else
        GRIDTOOLS_STATIC_ASSERT(NDim>0, "EYou are using a vector_accessor and compiling with C++03, switch to C++11 (-DENABLE_CXX11=ON)");
#endif
    };

    template <typename T>
    struct is_vector_accessor : boost::mpl::false_ {};

    template <uint_t ID, enumtype::intend Intent, typename Extent, uint_t Size >
    struct is_vector_accessor<vector_accessor<ID, Intent, Extent, Size> > : boost::mpl::true_ {};

    template <typename T>
    struct is_any_accessor : boost::mpl::or_<is_accessor<T>, is_vector_accessor<T> > {};

}//namespace gridtools
