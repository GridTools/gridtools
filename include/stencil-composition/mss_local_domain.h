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
#include "local_domain_metafunctions.h"

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

    template<
        enumtype::backend BackendId,
        typename MssType,
        typename DomainType,
        typename actual_arg_list_type,
        bool IsStateful
    >
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

        typedef typename boost::fusion::result_of::as_vector<mpl_local_domain_list>::type unfused_local_domain_list_t;

        typedef typename fuse_mss_local_domains<BackendId, unfused_local_domain_list_t>::type fused_local_domain_sequence_t;
        typedef typename generate_args_lookup_map<BackendId, unfused_local_domain_list_t, fused_local_domain_sequence_t>::type
                fused_local_domain_args_map;

        fused_local_domain_sequence_t local_domain_list;
    };

    template<typename T> struct is_mss_local_domain : boost::mpl::false_{};

    template<
        enumtype::backend BackendId,
        typename MssType,
        typename DomainType,
        typename actual_arg_list_type,
        bool IsStateful
    >
    struct is_mss_local_domain<mss_local_domain<BackendId, MssType, DomainType, actual_arg_list_type, IsStateful> > :
        boost::mpl::true_{};

    template<typename T>
    struct mss_local_domain_list
    {
        BOOST_STATIC_ASSERT((is_mss_local_domain<T>::value));
        typedef typename T::fused_local_domain_sequence_t type;
    };

    template<typename T>
    struct mss_local_domain_esf_args_map
    {
        BOOST_STATIC_ASSERT((is_mss_local_domain<T>::value));
        typedef typename T::fused_local_domain_args_map type;
    };

} //namespace gridtools
