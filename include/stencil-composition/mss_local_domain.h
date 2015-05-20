/*
 * mss_local_domain.h
 *
 *  Created on: Feb 18, 2015
 *      Author: carlosos
 */

#pragma once
#include "local_domain.h"

namespace gridtools
{
    namespace _impl{
        /*
         *
         * @name Few short and obvious metafunctions
         * @{
         * */
        template <typename StoragePointers, template <class , class , bool > class LocalDomain, bool IsStateful>
        struct get_local_domain {
            template <typename T>
            struct apply {
                typedef LocalDomain<StoragePointers,T,IsStateful> type;
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
            _impl::get_local_domain<actual_arg_list_type, local_domain, IsStateful>
            >::type mpl_local_domain_list;

        /**
         *
         */
        typedef typename boost::fusion::result_of::as_vector<mpl_local_domain_list>::type LocalDomainList;

        /**
         *
         */
        LocalDomainList local_domain_list;

    };
    template<typename T> struct is_mss_local_domain : boost::mpl::false_{};

    template<typename MssType, typename DomainType, typename actual_arg_list_type, bool IsStateful>
    struct is_mss_local_domain<mss_local_domain<MssType, DomainType, actual_arg_list_type, IsStateful> > : boost::mpl::true_{};

} //namespace gridtools
