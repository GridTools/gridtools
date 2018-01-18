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
#pragma once

#include <iostream>

#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/at_key.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/type_index.hpp>
#include <boost/type_traits/remove_pointer.hpp>

#include "../gridtools.hpp"
#include "../common/vector_traits.hpp"
#include "../common/gt_assert.hpp"
#include "accessor.hpp"
#include "arg.hpp"
#include "../storage/storage-facility.hpp"

namespace gridtools {
    namespace _impl {

        // metafunction that checks if argument is a suitable aggregator element (only arg_storage_pair)
        template < typename... Args >
        struct aggregator_arg_storage_pair_check;

        template < typename ArgStoragePair, typename... Rest >
        struct aggregator_arg_storage_pair_check< ArgStoragePair, Rest... > {
            typedef typename boost::decay< ArgStoragePair >::type arg_storage_pair_t;
            typedef typename is_arg_storage_pair< arg_storage_pair_t >::type is_suitable;
            typedef typename boost::mpl::and_< is_suitable,
                typename aggregator_arg_storage_pair_check< Rest... >::type > type;
        };

        template <>
        struct aggregator_arg_storage_pair_check<> {
            typedef boost::mpl::true_ type;
        };

        // metafunction that checks if argument is a suitable aggregator element (only data_store, data_store_field,
        // std::vector)
        template < typename... Args >
        struct aggregator_storage_check;

        template < typename DataStoreType, typename... Rest >
        struct aggregator_storage_check< DataStoreType, Rest... > {
            typedef typename boost::decay< DataStoreType >::type data_store_t;
            typedef typename is_data_store< data_store_t >::type c1;
            typedef typename is_data_store_field< data_store_t >::type c2;
            typedef typename is_vector< data_store_t >::type c3;
            typedef typename boost::mpl::or_< c1, c2, c3 >::type is_suitable;
            typedef typename boost::mpl::and_< is_suitable, typename aggregator_storage_check< Rest... >::type > type;
        };

        template <>
        struct aggregator_storage_check<> {
            typedef boost::mpl::true_ type;
        };

        struct l_get_arg_storage_pair_type {
            template < typename Arg >
            struct apply {
                typedef arg_storage_pair< Arg, typename Arg::data_store_t > type;
            };
        };

        template < typename Arg >
        struct create_arg_storage_pair_type {
            GRIDTOOLS_STATIC_ASSERT((is_arg< Arg >::value), GT_INTERNAL_ERROR_MSG("The given type is not an arg type"));
            typedef arg_storage_pair< Arg, typename Arg::data_store_t > type;
        };

        /** Metafunction class.
         *  This class is filling a fusion::vector of pointers to storages with pointers from given arg_storage_pairs
         */
        template < typename FusionVector >
        struct fill_arg_storage_pair_list {
            FusionVector &m_fusion_vec;

            template < typename ArgStoragePair,
                typename boost::enable_if< is_arg_storage_pair< typename std::decay< ArgStoragePair >::type >,
                    int >::type = 0 >
            void operator()(ArgStoragePair &&arg_storage_pair) const {
                boost::fusion::at_key< typename std::decay< ArgStoragePair >::type >(m_fusion_vec) =
                    std::forward< ArgStoragePair >(arg_storage_pair);
            }

            // reset fusion::vector of pointers to storages with fresh pointers from given storages (either data_store,
            // data_store_field, std::vector)
            template < typename ArgStoragePair, typename... Rest >
            void reassign(ArgStoragePair &&first, Rest &&... pairs) {
                (*this)(std::forward< ArgStoragePair >(first));
                reassign(std::forward< Rest >(pairs)...);
            }
            void reassign() {}
        };

        struct private_ctor_t {};
        constexpr private_ctor_t private_ctor{};
    } // namespace _impl

} // namespace gridtoold
