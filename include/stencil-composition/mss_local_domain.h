/*
 * mss_local_domain.h
 *
 *  Created on: Feb 18, 2015
 *      Author: carlosos
 */

#pragma once
#include <boost/mpl/at.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/has_key.hpp>

#include "local_domain.h"
#include "general_metafunctions.h"
#include "backend_traits_fwd.h"

namespace gridtools
{
    namespace _impl{
        /*
         *
         * @name Few short and obvious metafunctions
         * @{
         * */
        template <typename StoragePointers, bool IsStateful>
        struct get_local_domain {
            template <typename Esf>
            struct apply {
                BOOST_STATIC_ASSERT((is_esf_descriptor<Esf>::value));
                typedef local_domain<StoragePointers,typename Esf::args,IsStateful> type;
            };
        };
    } //namespace _impl

    template<typename MssType, typename DomainType, typename actual_arg_list_type, bool IsStateful>
    struct mss_local_domain
    {
        /**
         * Create a fusion::vector of domains for each functor
         *
         */
        typedef typename boost::mpl::transform<
            typename MssType::linear_esf,
            _impl::get_local_domain<actual_arg_list_type, IsStateful>
        >::type mpl_local_domain_list;

        typedef typename boost::fusion::result_of::as_vector<mpl_local_domain_list>::type LocalDomainList;

        LocalDomainList local_domain_list;
    };

    template<typename T> struct is_mss_local_domain : boost::mpl::false_{};

    template<typename MssType, typename DomainType, typename actual_arg_list_type, bool IsStateful>
    struct is_mss_local_domain<mss_local_domain<MssType, DomainType, actual_arg_list_type, IsStateful> > : boost::mpl::true_{};

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

    template<typename MssLocalDomain>
    struct create_trivial_args_lookup_map
    {
        BOOST_STATIC_ASSERT((is_mss_local_domain<MssLocalDomain>::value));
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
            typename MssLocalDomain::LocalDomainList,
            boost::mpl::vector0<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                generate_trivial_esf_args_map<boost::mpl::_2>
            >
        >::type type;
    };

    template<typename MssLocalDomain, typename MergedLocalDomainSequence>
    struct create_args_lookup_map
    {
        BOOST_STATIC_ASSERT((is_mss_local_domain<MssLocalDomain>::value));
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
            typename MssLocalDomain::LocalDomainList,
            boost::mpl::vector0<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                generate_esf_args_map<boost::mpl::_2>
            >
        >::type type;
    };

    template<enumtype::backend BackendId, typename MssLocalDomain>
    struct fuse_mss_local_domains
    {
        BOOST_STATIC_ASSERT((is_mss_local_domain<MssLocalDomain>::value));
        typedef typename boost::mpl::eval_if<
            typename backend_traits_from_id<BackendId>::mss_fuse_esfs_strategy,
            merge_local_domain_sequence<typename MssLocalDomain::LocalDomainList>,
            boost::mpl::identity<typename MssLocalDomain::LocalDomainList>
        >::type type;
    };

    template<enumtype::backend BackendId, typename MssLocalDomain, typename MergedLocalDomainSequence>
    struct generate_args_lookup_map
    {
        BOOST_STATIC_ASSERT((is_mss_local_domain<MssLocalDomain>::value));

        typedef typename boost::mpl::eval_if<
            typename backend_traits_from_id<BackendId>::mss_fuse_esfs_strategy,
            create_args_lookup_map<MssLocalDomain, MergedLocalDomainSequence>,
            create_trivial_args_lookup_map<MssLocalDomain>
        >::type type;
    };

} //namespace gridtools
