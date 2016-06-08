#pragma once
/** @file vector accessor */

namespace gridtools {

    /**
       @brief accessor for an expandable parameters list

       accessor object used with the expandable parameters. It is exactly like a regular accessor.
       Its type must be different though, so that the gridtools::iterate_domain can implement a specific
       overload of the operator() for this accessor type.

       \tparam ID integer identifier, to univocally specify the accessor
       \tparam Intent flag stating wether or not this accessor is read only
       \tparam Extent specification of the minimum box containing the stencil access pattern
       \tparam NDim dimensionality of the vector accessor: should be the storage space dimensions plus one (the vector
       field dimension)
    */
    template < uint_t ID, enumtype::intend Intent, typename LocationType, typename Extent >
    struct vector_accessor : accessor< ID, Intent, LocationType, Extent > {

        using super = accessor< ID, Intent, LocationType, Extent >;
        using super::accessor;
    };

    template < typename T >
    struct is_vector_accessor : boost::mpl::false_ {};

    template < uint_t ID, enumtype::intend Intent, typename LocationType, typename Extent >
    struct is_vector_accessor< vector_accessor< ID, Intent, LocationType, Extent > > : boost::mpl::true_ {};

    template < typename T >
    struct is_any_accessor : boost::mpl::or_< is_accessor< T >, is_vector_accessor< T > > {};

} // namespace gridtools
