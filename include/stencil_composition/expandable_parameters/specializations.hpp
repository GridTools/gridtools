namespace gridtools {
    // metafunction to access the storage type given the arg
    template < ushort_t ID, typename T >
    struct arg2storage< arg< ID, std::vector< pointer< T > > > > {
        typedef T type;
    };

    /** metafunction extracting the location type from the storage*/
    template < typename T >
    struct get_location_type< std::vector< T > > {
        typedef typename T::value_type::storage_info_type::index_type type;
    };

#ifdef CXX11_ENABLED
    template < typename Sequence, typename Arg >
    struct insert_if_not_present< Sequence, std::vector< pointer< Arg > > > : insert_if_not_present< Sequence, Arg > {
        using insert_if_not_present< Sequence, Arg >::insert_if_not_present;
    };

    /**
       specialization for expandable parameters
     */
    template < typename T >
    struct storage_holds_data_field< std::vector< pointer< T > > > : boost::mpl::true_ {};

    template < typename T >
    struct is_actual_storage< pointer< std::vector< pointer< T > > > > : public boost::mpl::bool_< !T::is_temporary > {
    };

    template < typename Storage >
    struct is_storage< std::vector< pointer< Storage > > > : is_storage< Storage > {};

    template < typename T >
    struct is_temporary_storage< std::vector< pointer< T > > > : public is_temporary_storage< T > {};

    template < typename T >
    struct is_any_storage< std::vector< T > > : is_any_storage< T > {};
#endif

#ifdef CXX11_ENABLED
    template < typename T, uint_t ID >
    struct is_actual_storage< pointer< storage<expandable_parameters< T, ID > > > >
        : public boost::mpl::bool_< !T::is_temporary > {};

    template < typename T, ushort_t Dim >
    struct is_temporary_storage< storage<expandable_parameters< T, Dim > > > : public boost::mpl::bool_< T::is_temporary > {};
#endif

} // namespace gridtools
