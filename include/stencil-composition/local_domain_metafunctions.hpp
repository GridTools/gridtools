/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include <boost/mpl/zip_view.hpp>

#include "local_domain.hpp"

namespace gridtools {

    struct get_storage_wrapper {
        template < typename LocalDomain >
        struct apply {
            typedef typename LocalDomain::storage_wrapper_list_t type;
        };
    };

    /**
     * @brief metafunction that merges a sequence of local domains into a single local domain
     */
    template < typename LocalDomainSequence, typename MssComponents, bool IsStateful >
    struct merge_local_domain_sequence {
        GRIDTOOLS_STATIC_ASSERT(
            (is_sequence_of< LocalDomainSequence, is_local_domain >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< LocalDomainSequence >::value > 0), "Internal Error: wrong size");

        // get extents map from first element
        typedef typename boost::mpl::at_c< LocalDomainSequence, 0 >::type::extents_map_t extents_map_t;
        // get all the storage wrapper lists from all local domains
        typedef typename boost::mpl::transform< LocalDomainSequence, get_storage_wrapper >::type all_storage_wrappers;
        // flatten the vector of vectors to a single vector
        typedef typename boost::mpl::fold< all_storage_wrappers,
            boost::mpl::vector0<>,
            boost::mpl::joint_view< boost::mpl::_1, boost::mpl::_2 > >::type all_storage_wrappers_flattened;
        // filter out duplicates
        typedef typename boost::mpl::fold< all_storage_wrappers_flattened,
            boost::mpl::vector0<>,
            boost::mpl::if_< boost::mpl::contains< boost::mpl::_1, boost::mpl::_2 >,
                                               boost::mpl::_1,
                                               boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 > > >::type
            merged_storage_wrappers;
        // retrieve all the args
        typedef typename boost::mpl::fold< merged_storage_wrappers,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1, get_arg_from_storage_wrapper< boost::mpl::_2 > > >::type new_args_t;

        // return the new local domain type
        typedef boost::mpl::vector< local_domain< merged_storage_wrappers, new_args_t, extents_map_t, IsStateful > >
            type;
    };

    /**
     * @brief metafunction that creates a trivial lookup map for the arguments (i.e. the identity)
     * from original positions in the local domain into the merge local domain. This trivial map
     * is used when no merging is required.
     */
    template < typename LocalDomainSequence >
    struct create_trivial_args_lookup_map {
        GRIDTOOLS_STATIC_ASSERT(
            (is_sequence_of< LocalDomainSequence, is_local_domain >::value), "Internal Error: wrong type");
        template < typename LocalDomain >
        struct generate_trivial_esf_args_map {
            GRIDTOOLS_STATIC_ASSERT((is_local_domain< LocalDomain >::value), "Internal Error: wrong type");
            typedef typename boost::mpl::fold< typename local_domain_esf_args< LocalDomain >::type,
                boost::mpl::map0<>,
                boost::mpl::insert< boost::mpl::_1, boost::mpl::pair< boost::mpl::_2, boost::mpl::_2 > > >::type type;
        };
        typedef typename boost::mpl::fold< LocalDomainSequence,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1, generate_trivial_esf_args_map< boost::mpl::_2 > > >::type type;
    };

    /**
     * @brief metafunction that generates the lookup table of the arguments that map their indices
     * in the original local domains into their positions in the merge local domain
     * @tparam LocalDomainSequence original sequence of local domains
     * @tparam MergedLocalDomainSequence merge local domain sequence
     */
    template < typename LocalDomainSequence, typename MergedLocalDomainSequence >
    struct create_args_lookup_map {
        GRIDTOOLS_STATIC_ASSERT(
            (is_sequence_of< LocalDomainSequence, is_local_domain >::value), "Internal Error: wrong type");
        // a real merged local domain should have only one element in the sequence
        // (as all the local domains were merged)
        GRIDTOOLS_STATIC_ASSERT(
            (boost::mpl::size< MergedLocalDomainSequence >::value == 1), "Internal Error: wrong size");
        typedef typename boost::mpl::front< MergedLocalDomainSequence >::type merged_local_domain_t;
        typedef typename local_domain_esf_args< merged_local_domain_t >::type merged_esf_args_t;

        template < typename Arg >
        struct find_arg_position_in_merged_domain {
            typedef typename boost::mpl::find< merged_esf_args_t, Arg >::type pos;
            GRIDTOOLS_STATIC_ASSERT((!boost::is_same< pos, merged_esf_args_t >::value), "Internal Error: wrong type");

            typedef
                typename boost::mpl::distance< typename boost::mpl::begin< merged_esf_args_t >::type, pos >::type type;
            BOOST_STATIC_CONSTANT(int, value = (type::value));
        };

        template < typename LocalDomain >
        struct generate_esf_args_map {
            typedef typename local_domain_esf_args< LocalDomain >::type local_domain_esf_args_t;
            GRIDTOOLS_STATIC_ASSERT((is_local_domain< LocalDomain >::value), "Internal Error: wrong type");
            typedef typename boost::mpl::fold<
                boost::mpl::zip_view< boost::mpl::vector2< local_domain_esf_args_t,
                    boost::mpl::range_c< int, 0, boost::mpl::size< local_domain_esf_args_t >::value > > >,
                boost::mpl::map0<>,
                boost::mpl::insert<
                    boost::mpl::_1,
                    boost::mpl::pair< boost::mpl::back< boost::mpl::_2 >,
                        find_arg_position_in_merged_domain< boost::mpl::front< boost::mpl::_2 > > > > >::type type;
        };

        typedef typename boost::mpl::fold< LocalDomainSequence,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1, generate_esf_args_map< boost::mpl::_2 > > >::type type;
    };

    /**
     * @brief metafunction that fuses the local domains of a mss if the backend requires it
     * @tparam BackendId id of the backend
     * @tparam LocalDomainSequence sequence of local domains
     */
    template < enumtype::platform BackendId, typename LocalDomainSequence, typename MssComponents, bool IsStateful >
    struct fuse_mss_local_domains {
        GRIDTOOLS_STATIC_ASSERT(
            (is_sequence_of< LocalDomainSequence, is_local_domain >::value), "Internal Error: wrong type");
        typedef typename boost::mpl::eval_if< typename backend_traits_from_id< BackendId >::mss_fuse_esfs_strategy,
            merge_local_domain_sequence< LocalDomainSequence, MssComponents, IsStateful >,
            boost::mpl::identity< LocalDomainSequence > >::type fused_mss_local_domains_t;

        typedef typename boost::fusion::result_of::as_vector< fused_mss_local_domains_t >::type type;
    };

    /**
     * @brief metafunction that generates the lookup map of arguments for arguments from the original
     * local domains positions into the fused local domain
     * @tparam BackendId id of the backend
     * @tparam LocalDomainSequence sequence of local domains
     * @tparam MergedLocalDomainSequence sequence of merged local domains
     */
    template < enumtype::platform BackendId, typename LocalDomainSequence, typename MergedLocalDomainSequence >
    struct generate_args_lookup_map {
        GRIDTOOLS_STATIC_ASSERT(
            (is_sequence_of< LocalDomainSequence, is_local_domain >::value), "Internal Error: wrong type");

        typedef typename boost::mpl::eval_if< typename backend_traits_from_id< BackendId >::mss_fuse_esfs_strategy,
            create_args_lookup_map< LocalDomainSequence, MergedLocalDomainSequence >,
            create_trivial_args_lookup_map< LocalDomainSequence > >::type type;
    };

    /**
     * @brief metafunction that computes the index of the position of a local domain in a fused local domain
     * @tparam Index original position index of the local domain in the non fused sequence
     * @tparam BackendId id of the backend
     */
    template < typename Index, typename BackendId >
    struct extract_local_domain_index {
        typedef typename boost::mpl::if_< typename backend_traits_from_id< BackendId::value >::mss_fuse_esfs_strategy,
            boost::mpl::int_< 0 >,
            Index >::type type;
    };
} // gridtools
