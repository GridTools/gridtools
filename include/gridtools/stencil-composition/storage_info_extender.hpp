/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../common/layout_map_metafunctions.hpp"
#include "../storage/storage-facility.hpp"

namespace gridtools {

    /**
     * @brief The extend_aux_param struct
     * it extends the declaration of a template parameter used by StorageInfo by NExtraDim dimensions
     */
    // specialization for parameters that are dimension independent, the metafunction has no impact
    template <ushort_t NExtraDim, typename T>
    struct extend_aux_param {
        typedef T type;
    };

    // specialization for a halo template parameter
    template <ushort_t NExtraDim, uint_t... Args>
    struct extend_aux_param<NExtraDim, halo<Args...>> {
        typedef typename repeat_template_c<0, NExtraDim, halo, Args...>::type type;
    };

    template <typename T, ushort_t NExtraDim>
    struct storage_info_extender_impl;

    template <template <typename...> class Base, typename First, ushort_t NExtraDim, typename... TmpParam>
    struct storage_info_extender_impl<Base<First, TmpParam...>, NExtraDim> {
        typedef Base<typename storage_info_extender_impl<First, NExtraDim>::type,
            typename extend_aux_param<NExtraDim, TmpParam>::type...>
            type;
    };

    template <int_t Val, short_t NExtraDim>
    struct inc_ {
        static const int_t value = Val == -1 ? -1 : Val + NExtraDim;
    };

    template <ushort_t NExtraDim, int_t... Args>
    struct storage_info_extender_impl<layout_map<Args...>, NExtraDim> {
        using type = typename extend_layout_map<layout_map<Args...>, NExtraDim>::type;
    };

    template <template <unsigned, typename, typename, typename> class StorageInfo,
        unsigned Index,
        typename Layout,
        typename Halo,
        typename Alignment,
        ushort_t NExtraDim>
    struct storage_info_extender_impl<StorageInfo<Index, Layout, Halo, Alignment>, NExtraDim> {
        typedef typename extend_aux_param<NExtraDim, Halo>::type new_halo_t;
        typedef StorageInfo<Index, typename storage_info_extender_impl<Layout, NExtraDim>::type, new_halo_t, Alignment>
            type;
    };

    // new storage info instantiation mechanism
    template <uint_t N, uint_t M, typename StorageInfo, typename OldStorageInfo, typename... Args>
    static constexpr typename boost::enable_if_c<(N > 0 && M > 0), StorageInfo>::type get_storage_info_instance(
        OldStorageInfo const &os, int extradim_length, Args... args) {
        return get_storage_info_instance<N, M - 1, StorageInfo>(
            os, extradim_length, args..., os.template total_length<OldStorageInfo::layout_t::masked_length - M>());
    }

    template <uint_t N, uint_t M, typename StorageInfo, typename OldStorageInfo, typename... Args>
    static constexpr typename boost::enable_if_c<(N > 0 && M == 0), StorageInfo>::type get_storage_info_instance(
        OldStorageInfo const &os, int extradim_length, Args... args) {
        return get_storage_info_instance<N - 1, 0, StorageInfo>(os, extradim_length, args..., extradim_length);
    }

    template <uint_t N, uint_t M, typename StorageInfo, typename OldStorageInfo, typename... Args>
    static constexpr typename boost::enable_if_c<(N == 0 && M == 0), StorageInfo>::type get_storage_info_instance(
        OldStorageInfo const &os, int extradim_length, Args... args) {
        return StorageInfo(args...);
    }

    // retrieve functions
    template <uint_t N, typename StorageInfo, typename R = typename storage_info_extender_impl<StorageInfo, N>::type>
    static constexpr R by(StorageInfo const &other, int extradim_length) {
        return get_storage_info_instance<N, StorageInfo::layout_t::masked_length, R>(other, extradim_length);
    }

    /**
     * @brief The storage_info_extender struct
     * helper that extends a storage_info by certain number of dimensions. Lengths of the extra dimensions are passed by
     * arguments. Values of halos of extra dims are set to null, and the layout of the new meta storage is such that the
     * newly added dimensions have the largest stride.
     NOTE: the extended meta_storage in not a literal type, while the storage_info is
     */
    struct storage_info_extender {

        template <typename StorageInfoPtr,
            typename StorageInfo = typename boost::remove_cv<typename StorageInfoPtr::element_type>::type>
        typename storage_info_extender_impl<StorageInfo, 1>::type operator()(
            StorageInfoPtr other, uint_t extradim_length) const {
            GT_STATIC_ASSERT((is_storage_info<StorageInfo>::value), "Use with a StorageInfo type only");
            typedef typename storage_info_extender_impl<StorageInfo, 1>::type type;

            return by<1>(*other, extradim_length);
        }
    };
} // namespace gridtools
