#pragma once
namespace gridtools {
    template < typename T >
    struct is_fusion_vector : boost::mpl::false_ {};

    template < typename... T >
    struct is_fusion_vector< boost::fusion::vector< T... > > : boost::mpl::true_ {};

} // namespace gridtools
