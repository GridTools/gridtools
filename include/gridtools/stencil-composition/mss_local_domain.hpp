/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
/*
 * mss_local_domain.h
 *
 *  Created on: Feb 18, 2015
 *      Author: carlosos
 */

#pragma once
#include <boost/mpl/at.hpp>
#include <boost/mpl/has_key.hpp>
#include <boost/mpl/set.hpp>

#include "aggregator_type.hpp"
#include "backend_traits_fwd.hpp"
#include "local_domain.hpp"
#include "local_domain_metafunctions.hpp"
#include "mss_components.hpp"
#include "storage_wrapper.hpp"

namespace gridtools {
    namespace _impl {
        /*
         *
         * @name Few short and obvious metafunctions
         * @{
         * */
        template < typename StorageWrapperList, bool IsStateful >
        struct get_local_domain {
            template < typename Esf >
            struct apply {
                GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< Esf >::value), GT_INTERNAL_ERROR);
                // filter out the view wrappers that are used by the esf
                typedef
                    typename boost::mpl::fold< typename Esf::args_t,
                        boost::mpl::vector0<>,
                        boost::mpl::push_back< boost::mpl::_1,
                                                   storage_wrapper_elem< boost::mpl::_2, StorageWrapperList > > >::type
                        local_view_wrapper_list;

                // create a local_domain type specialized with the  local view wrapper list and a
                // single Esf (vector is needed because local_domains (esfs) might be fusioned later on)
                typedef local_domain< local_view_wrapper_list, typename Esf::args_t, IsStateful > type;
            };
        };
    } // namespace _impl

    template < enumtype::platform BackendId, typename MssComponents, typename StorageWrapperList, bool IsStateful >
    struct mss_local_domain {
        GRIDTOOLS_STATIC_ASSERT((is_mss_components< MssComponents >::value), GT_INTERNAL_ERROR);

        /**
         * Create a fusion::vector of domains for each functor
         *
         */
        typedef typename boost::mpl::transform< typename MssComponents::linear_esf_t,
            _impl::get_local_domain< StorageWrapperList, IsStateful > >::type mpl_local_domain_list;

        typedef
            typename boost::fusion::result_of::as_vector< mpl_local_domain_list >::type unfused_local_domain_sequence_t;

        typedef typename fuse_mss_local_domains< BackendId, mpl_local_domain_list, MssComponents, IsStateful >::type
            fused_local_domain_sequence_t;

        typedef typename generate_args_lookup_map< BackendId,
            unfused_local_domain_sequence_t,
            fused_local_domain_sequence_t >::type fused_local_domain_args_map;

        fused_local_domain_sequence_t local_domain_list;
    };

    template < typename T >
    struct is_mss_local_domain : boost::mpl::false_ {};

    template < enumtype::platform BackendId, typename MssType, typename StorageWrapperList, bool IsStateful >
    struct is_mss_local_domain< mss_local_domain< BackendId, MssType, StorageWrapperList, IsStateful > >
        : boost::mpl::true_ {};

    template < typename T >
    struct mss_local_domain_list {
        GRIDTOOLS_STATIC_ASSERT((is_mss_local_domain< T >::value), GT_INTERNAL_ERROR);
        typedef typename T::fused_local_domain_sequence_t type;
    };

    template < typename T >
    struct mss_local_domain_esf_args_map {
        GRIDTOOLS_STATIC_ASSERT((is_mss_local_domain< T >::value), GT_INTERNAL_ERROR);
        typedef typename T::fused_local_domain_args_map type;
    };

} // namespace gridtools
