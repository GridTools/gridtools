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

        template<typename set_state, typename esf_sequence>
        struct insert_new_esf_args
        {
            typedef typename boost::mpl::fold<
                esf_sequence,
                set_state,
                boost::mpl::insert<boost::mpl::_1,boost::mpl::_2>
            >::type type;
        };

        typedef typename boost::mpl::fold<
            LocalDomainSequence,
            boost::mpl::set0<>,
            insert_new_esf_args<boost::mpl::_1, local_domain_esf_args<boost::mpl::_2> >
        >::type merged_esf_args_t;

        typedef local_domain<
            typename local_domain_storage_pointers<typename boost::mpl::front<LocalDomainSequence>::type>::type,
            merged_esf_args_t,
            local_domain_is_stateful< typename boost::mpl::front<LocalDomainSequence>::type >::value
        > type;
    };

    template<typename MssLocalDomain>
    struct create_trivial_args_lookup_map
    {
        template<typename EsfSequence>
        struct generate_trivial_esf_args_map
        {
            typedef typename boost::mpl::fold<
                EsfSequence,
                boost::mpl::map0<>,
                boost::mpl::insert<
                    boost::mpl::_1,
                    lazy_pair<boost::mpl::_2, boost::mpl::_2>
                >
            >::type type;
        };
        typedef typename boost::mpl::fold<
            typename local_domain_esf_args<typename MssLocalDomain::LocalDomainList>::type,
            boost::mpl::vector0<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                generate_trivial_esf_args_map<boost::mpl::_2>
            >
        >::type type;
    };

} //namespace gridtools
