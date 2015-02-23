/*
 * mss_local_domain.h
 *
 *  Created on: Feb 18, 2015
 *      Author: carlosos
 */

#pragma once

namespace gridtools
{
    namespace _impl{
        /*
         *
         * @name Few short and obvious metafunctions
         * @{
         * */
        template <typename StoragePointers, typename Iterators, template <class A, class B, class C> class LocalDomain>
        struct get_local_domain {
            template <typename T>
            struct apply {
                typedef LocalDomain<StoragePointers,Iterators,T> type;
            };
        };
    } //namespace _impl

    template<typename MssType, typename DomainType, typename actual_arg_list_type>
    struct mss_local_domain
    {
        /**
         * Create a fusion::vector of domains for each functor
         *
         */
        typedef typename boost::mpl::transform<
            typename MssType::linear_esf,
            _impl::get_local_domain<actual_arg_list_type, typename DomainType::iterator_list, local_domain>
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
} //namespace gridtools
