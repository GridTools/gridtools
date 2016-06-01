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

#include "aggregator_type.hpp"
#include "local_domain.hpp"
#include "backend_traits_fwd.hpp"
#include "mss_components.hpp"
#include "local_domain_metafunctions.hpp"

namespace gridtools {
    namespace _impl {
        /*
         *
         * @name Few short and obvious metafunctions
         * @{
         * */
        template < typename StoragePointers, typename MetaStoragePointers, bool IsStateful >
        struct get_local_domain {
            template < typename Esf >
            struct apply {
                GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< Esf >::value), "Internal Error: invalid type");
                typedef local_domain< StoragePointers, MetaStoragePointers, typename Esf::args_t, IsStateful > type;
            };
        };
    } // namespace _impl

    template < enumtype::platform BackendId,
        typename MssComponents,
        typename DomainType,
        typename ActualArgListType,
        typename MetaStorageListType,
        bool IsStateful >
    struct mss_local_domain {
        GRIDTOOLS_STATIC_ASSERT((is_mss_components< MssComponents >::value), "Internal Error: invalid type");
        GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< DomainType >::value), "Internal Error: invalid type");

        /**
         * Create a fusion::vector of domains for each functor
         *
         */
        typedef typename boost::mpl::transform< typename MssComponents::linear_esf_t,
            _impl::get_local_domain< ActualArgListType, MetaStorageListType, IsStateful > >::type mpl_local_domain_list;

        typedef
            typename boost::fusion::result_of::as_vector< mpl_local_domain_list >::type unfused_local_domain_sequence_t;

        typedef typename fuse_mss_local_domains< BackendId, unfused_local_domain_sequence_t >::type
            fused_local_domain_sequence_t;
        typedef typename generate_args_lookup_map< BackendId,
            unfused_local_domain_sequence_t,
            fused_local_domain_sequence_t >::type fused_local_domain_args_map;

        fused_local_domain_sequence_t local_domain_list;
    };

    template < typename T >
    struct is_mss_local_domain : boost::mpl::false_ {};

    template < enumtype::platform BackendId,
        typename MssType,
        typename DomainType,
        typename ActualArgListType,
        typename MetaStorageListType,
        bool IsStateful >
    struct is_mss_local_domain<
        mss_local_domain< BackendId, MssType, DomainType, ActualArgListType, MetaStorageListType, IsStateful > >
        : boost::mpl::true_ {};

    template < typename T >
    struct mss_local_domain_list {
        GRIDTOOLS_STATIC_ASSERT((is_mss_local_domain< T >::value), "Internal Error: invalid type");
        typedef typename T::fused_local_domain_sequence_t type;
    };

    template < typename T >
    struct mss_local_domain_esf_args_map {
        GRIDTOOLS_STATIC_ASSERT((is_mss_local_domain< T >::value), "Internal Error: invalid type");
        typedef typename T::fused_local_domain_args_map type;
    };

} // namespace gridtools
