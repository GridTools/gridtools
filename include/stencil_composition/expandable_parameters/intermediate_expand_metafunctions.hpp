#pragma once

/** @file metafunctions used in @ref gridtools::intermediate_expand*/

namespace gridtools {
    namespace _impl {

        // ********* metafunctions ************
        template < typename T >
        struct is_expandable_parameters : boost::mpl::false_ {};

        template < typename BaseStorage, ushort_t N >
        struct is_expandable_parameters< expandable_parameters< BaseStorage, N > > : boost::mpl::true_ {};

        template < typename BaseStorage >
        struct is_expandable_parameters< std::vector< pointer< BaseStorage > > > : boost::mpl::true_ {};

        template < typename T >
        struct is_expandable_arg : boost::mpl::false_ {};

        template < uint_t N, typename Storage, typename Condition >
        struct is_expandable_arg< arg< N, Storage, Condition > > : is_expandable_parameters< Storage > {};

        template < uint_t N, typename Storage >
        struct is_expandable_arg< arg< N, Storage > > : is_expandable_parameters< Storage > {};

        template < typename T >
        struct get_basic_storage {
            typedef typename T::storage_type::basic_type type;
        };

        template < ushort_t ID, typename T >
        struct get_basic_storage< arg< ID, std::vector< pointer< T > > > > {
            typedef typename T::basic_type type;
        };

        template < typename T >
        struct get_storage {
            typedef typename T::storage_type type;
        };

        template < typename T >
        struct get_index {
            typedef typename T::index_type type;
            static const uint_t value = T::index_type::value;
        };

        template < enumtype::platform B >
        struct create_arg;

        template <>
        struct create_arg< enumtype::Host > {
            template < typename T, typename ExpandFactor >
            struct apply {
                typedef arg< get_index< T >::value,
                             storage<expandable_parameters< typename get_basic_storage< T >::type, ExpandFactor::value> > > type;
            };

            template < typename T, typename ExpandFactor, uint_t ID >
            struct apply< arg< ID, std::vector< pointer< no_storage_type_yet< T > > > >, ExpandFactor > {
                typedef arg< ID,
                             no_storage_type_yet<  storage<expandable_parameters<typename T::basic_type, ExpandFactor::value > > > > type;
            };
        };

        template <>
        struct create_arg< enumtype::Cuda > {
            template < typename T, typename ExpandFactor >
            struct apply {
                typedef arg< get_index< T >::value,
                             storage< expandable_parameters< typename get_basic_storage< T >::type, ExpandFactor::value > > >
                    type;
            };

            template < uint_t ID, typename T, typename ExpandFactor >
            struct apply< arg< ID, std::vector< pointer< no_storage_type_yet< T > > > >, ExpandFactor > {
                typedef arg< ID,
                    no_storage_type_yet< storage<
                                             expandable_parameters< typename T::basic_type, ExpandFactor::value> > > > type;
            };
        };

    } // namespace _impl
} // namespace gridtools
