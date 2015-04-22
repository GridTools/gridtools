#pragma once

namespace gridtools {

    template<typename LocalDomainSequence>
    struct merge_local_domain_sequence
    {
        BOOST_STATIC_ASSERT((is_sequence_of<LocalDomainSequence, is_local_domain>::value));
        BOOST_STATIC_ASSERT((boost::mpl::size<LocalDomainSequence>::value > 0));

        template<typename state, typename element>
        struct insert_element
        {
            typedef boost::mpl::pair<
                typename boost::mpl::insert<typename boost::mpl::first<state>::type, element>::type,
                typename boost::mpl::push_back<typename boost::mpl::second<state>::type, element>::type
            > type;
        };

        template<typename state, typename esf_sequence>
        struct insert_new_esf_args
        {
            typedef typename boost::mpl::first<state>::type assoc_state_t;

            typedef typename boost::mpl::fold<
                esf_sequence,
                state,
                boost::mpl::if_<
                    boost::mpl::has_key<
                        assoc_state_t,
                        boost::mpl::_2
                    >,
                    boost::mpl::_1,
                    insert_element<boost::mpl::_1, boost::mpl::_2>
                >
            >::type type;
        };

        typedef typename boost::mpl::fold<
            LocalDomainSequence,
            boost::mpl::pair< boost::mpl::set0<>, boost::mpl::vector0<> >,
            insert_new_esf_args<boost::mpl::_1, local_domain_esf_args<boost::mpl::_2> >
        >::type merged_esf_args2_t;

        typedef typename boost::mpl::second<merged_esf_args2_t>::type merged_esf_args_t;

        typedef boost::mpl::vector1<
            local_domain<
                typename local_domain_storage_pointers<typename boost::mpl::front<LocalDomainSequence>::type>::type,
                merged_esf_args_t,
                local_domain_is_stateful< typename boost::mpl::front<LocalDomainSequence>::type >::value
            >
        > type;
    };

    template<typename LocalDomainSequence>
    struct create_trivial_args_lookup_map
    {
        BOOST_STATIC_ASSERT((is_sequence_of<LocalDomainSequence, is_local_domain>::value));
        template<typename LocalDomain>
        struct generate_trivial_esf_args_map
        {
            BOOST_STATIC_ASSERT((is_local_domain<LocalDomain>::value));
            typedef typename boost::mpl::fold<
                typename local_domain_esf_args<LocalDomain>::type,
                boost::mpl::map0<>,
                boost::mpl::insert<
                    boost::mpl::_1,
                    boost::mpl::pair<boost::mpl::_2, boost::mpl::_2>
                >
            >::type type;
        };
        typedef typename boost::mpl::fold<
            LocalDomainSequence,
            boost::mpl::vector0<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                generate_trivial_esf_args_map<boost::mpl::_2>
            >
        >::type type;
    };

    template<typename LocalDomainSequence, typename MergedLocalDomainSequence>
    struct create_args_lookup_map
    {
        BOOST_STATIC_ASSERT((is_sequence_of<LocalDomainSequence, is_local_domain>::value));
        // a real merged local domain should have only one element in the sequence
        // (as all the local domains were merged)
        BOOST_STATIC_ASSERT((boost::mpl::size<MergedLocalDomainSequence>::value==1));
        typedef typename boost::mpl::front<MergedLocalDomainSequence>::type merged_local_domain_t;
        typedef typename local_domain_esf_args<merged_local_domain_t>::type merged_esf_args_t;

        template<typename Arg>
        struct find_arg_position_in_merged_domain
        {
            typedef typename boost::mpl::find<
                merged_esf_args_t,
                Arg
            >::type pos;
            BOOST_STATIC_ASSERT((!boost::is_same<pos, merged_esf_args_t>::value));

            typedef typename boost::mpl::distance<
                typename boost::mpl::begin<merged_esf_args_t>::type,
                pos
            >::type type;
            BOOST_STATIC_CONSTANT(int, value = (type::value));
        };

        template<typename LocalDomain>
        struct generate_esf_args_map
        {
            typedef typename local_domain_esf_args<LocalDomain>::type local_domain_esf_args_t;
            BOOST_STATIC_ASSERT((is_local_domain<LocalDomain>::value));
            typedef typename boost::mpl::fold<
                boost::mpl::zip_view<
                    boost::mpl::vector2<
                        local_domain_esf_args_t,
                        boost::mpl::range_c<int, 0, boost::mpl::size<local_domain_esf_args_t>::value >
                    >
                >,
                boost::mpl::map0<>,
                boost::mpl::insert<
                    boost::mpl::_1,
                    boost::mpl::pair<
                        boost::mpl::back<boost::mpl::_2>,
                        find_arg_position_in_merged_domain<
                            boost::mpl::front<boost::mpl::_2>
                        >
                    >
                >
            >::type type;
        };

        typedef typename boost::mpl::fold<
            LocalDomainSequence,
            boost::mpl::vector0<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                generate_esf_args_map<boost::mpl::_2>
            >
        >::type type;
    };

    template<enumtype::backend BackendId, typename LocalDomainSequence>
    struct fuse_mss_local_domains
    {
        BOOST_STATIC_ASSERT((is_sequence_of<LocalDomainSequence, is_local_domain>::value));
        typedef typename boost::mpl::eval_if<
            typename backend_traits_from_id<BackendId>::mss_fuse_esfs_strategy,
            merge_local_domain_sequence<LocalDomainSequence>,
            boost::mpl::identity<LocalDomainSequence>
        >::type type;
    };

    template<enumtype::backend BackendId, typename LocalDomainSequence, typename MergedLocalDomainSequence>
    struct generate_args_lookup_map
    {
        BOOST_STATIC_ASSERT((is_sequence_of<LocalDomainSequence, is_local_domain>::value));

        typedef typename boost::mpl::eval_if<
            typename backend_traits_from_id<BackendId>::mss_fuse_esfs_strategy,
            create_args_lookup_map<LocalDomainSequence, MergedLocalDomainSequence>,
            create_trivial_args_lookup_map<LocalDomainSequence>
        >::type type;
    };
} // gridtools
