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

namespace gridtools {

    /**
     * @brief The storage_info_extender struct
     * helper that extends a metastorage by certain number of dimensions. Lengths of the extra dimensions are passed by
     * arguments. Values of halos of extra dims are set to null, and the layout of the new meta storage is such that the
     * newly added dimensions have the largest stride.
     NOTE: the extended meta_storage in not a literal type, while the storage_info is
     */
    struct storage_info_extender {
        // layout extender
        template < uint_t N, typename Layout >
        struct get_extended_layout_type;

        template < uint_t N, int... Ints >
        struct get_extended_layout_type< N, layout_map< Ints... > >
            : get_extended_layout_type< (N - 1), layout_map< layout_map< Ints... >::unmasked_length, Ints... > > {};

        template < int... Ints >
        struct get_extended_layout_type< 0, layout_map< Ints... > > {
            typedef layout_map< Ints... > type;
        };

        // halo extender
        template < uint_t N, typename Halo >
        struct get_extended_halo_type;

        template < uint_t N, unsigned... M >
        struct get_extended_halo_type< N, halo< M... > > : get_extended_halo_type< (N - 1), halo< M..., 0 > > {};

        template < unsigned... M >
        struct get_extended_halo_type< 0, halo< M... > > {
            typedef halo< M... > type;
        };

        // storage info extender
        template < uint_t N, typename StorageInfo >
        struct get_extended_storage_info_type;

        template < uint_t N,
            template < unsigned, typename, typename, typename > class StorageInfo,
            unsigned Index,
            typename Layout,
            typename Halo,
            typename Alignment >
        struct get_extended_storage_info_type< N, StorageInfo< Index, Layout, Halo, Alignment > > {
            GRIDTOOLS_STATIC_ASSERT((is_storage_info< StorageInfo< Index, Layout, Halo, Alignment > >::value),
                "Use with a StorageInfo type only");
            typedef typename get_extended_layout_type< N, Layout >::type ext_layout_t;
            typedef typename get_extended_halo_type< N, Halo >::type ext_halo_t;
            typedef StorageInfo< Index, ext_layout_t, ext_halo_t, Alignment > type;
        };

        // new storage info instantiation mechanism
        template < uint_t N, uint_t M, typename StorageInfo, typename OldStorageInfo, typename... Args >
        static constexpr typename boost::enable_if_c< (N > 0 && M > 0), StorageInfo >::type get_storage_info_instance(
            OldStorageInfo const &os, int extradim_length, Args... args) {
            return get_storage_info_instance< N, M - 1, StorageInfo >(
                os, extradim_length, args..., os.template dim< OldStorageInfo::layout_t::masked_length - M >());
        }

        template < uint_t N, uint_t M, typename StorageInfo, typename OldStorageInfo, typename... Args >
        static constexpr typename boost::enable_if_c< (N > 0 && M == 0), StorageInfo >::type get_storage_info_instance(
            OldStorageInfo const &os, int extradim_length, Args... args) {
            return get_storage_info_instance< N - 1, 0, StorageInfo >(os, extradim_length, args..., extradim_length);
        }

        template < uint_t N, uint_t M, typename StorageInfo, typename OldStorageInfo, typename... Args >
        static constexpr typename boost::enable_if_c< (N == 0 && M == 0), StorageInfo >::type get_storage_info_instance(
            OldStorageInfo const &os, int extradim_length, Args... args) {
            return StorageInfo(args...);
        }

        // retrieve functions
        template < uint_t N,
            typename StorageInfo,
            typename R = typename get_extended_storage_info_type< N, StorageInfo >::type >
        static constexpr R by(StorageInfo const &other, int extradim_length) {
            return get_storage_info_instance< N, StorageInfo::layout_t::masked_length, R >(other, extradim_length);
        }

        template < typename StorageInfo >
        constexpr typename get_extended_storage_info_type< 1, StorageInfo >::type operator()(
            StorageInfo const &other, int extradim_length) const {
            GRIDTOOLS_STATIC_ASSERT((is_storage_info< StorageInfo >::value), "Use with a StorageInfo type only");
            return by< 1 >(other, extradim_length);
        }

        template < typename StorageInfo >
        constexpr typename get_extended_storage_info_type< 1, StorageInfo >::type operator()(
            StorageInfo const *other, int extradim_length) const {
            GRIDTOOLS_STATIC_ASSERT((is_storage_info< StorageInfo >::value), "Use with a StorageInfo type only");
            return by< 1 >(*other, extradim_length);
        }
    };
}
