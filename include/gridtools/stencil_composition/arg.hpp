/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**
   @file
   @brief File containing the definition of the placeholders used to address the storage from whithin the functors.
   A placeholder is an implementation of the proxy design pattern for the storage class, i.e. it is a light object used
   in place of the storage when defining the high level computations,
   and it will be bound later on with a specific instantiation of a storage class.
*/

#pragma once

#include <type_traits>
#include <utility>
#include <vector>

#include "../common/defs.hpp"
#include "../storage/storage_facility.hpp"
#include "location_type.hpp"

namespace gridtools {

    template <class Tag, class DataStore, class Location, bool Temporary>
    struct plh;

    template <class>
    struct is_plh : std::false_type {};

    template <class Tag, class DataStore, class Location, bool Temporary>
    struct is_plh<plh<Tag, DataStore, Location, Temporary>> : std::true_type {};

    /** @brief binding between the placeholder (\tparam Plh) and the storage (\tparam DataStore)*/
    template <class Plh, class DataStore>
    struct arg_storage_pair {

        GT_STATIC_ASSERT(is_plh<Plh>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((std::is_same<typename Plh::data_store_t, std::decay_t<DataStore>>::value),
            "DataStoreType type not compatible with placeholder storage type, when associating placeholder to actual "
            "data store");

        static constexpr Plh arg() { return {}; }

        DataStore m_value;

        using arg_t = Plh;
        using data_store_t = std::remove_reference_t<DataStore>;
    };

    template <class>
    struct is_arg_storage_pair : std::false_type {};

    template <class Plh, class DataStore>
    struct is_arg_storage_pair<arg_storage_pair<Plh, DataStore>> : std::true_type {};

    template <class>
    struct is_tmp_arg : std::false_type {};

    template <class Tag, class DataStore, class Location>
    struct is_tmp_arg<plh<Tag, DataStore, Location, true>> : std::true_type {};

    template <class Plh, class DataStore>
    struct is_tmp_arg<arg_storage_pair<Plh, DataStore>> : is_tmp_arg<Plh> {};

    /**
     * Type to create placeholders for data fields.
     *
     * There is a specialization for the case in which T is a temporary.
     * The default version applies to all the storage classes (including
     * user-defined ones used via the global-accessor)
     *
     * @tparam I Integer index (unique) of the data field to identify it
     * @tparam DataStoreType The type of the storage used to store data
     * @tparam LocationType the location type of the storage of the placeholder
     * @tparam Temporary determines whether the placeholder holds a temporary or normal storage
     */
    template <class Tag, class DataStore, class Location, bool Temporary>
    struct plh {
        GT_STATIC_ASSERT(
            is_location_type<Location>::value, "The third template argument of a placeholder must be a location_type");
        using data_store_t = DataStore;
        using location_t = Location;
        using tag_t = Tag;

        template <class T>
        arg_storage_pair<plh, T> operator=(T &&arg) const {
            return {std::forward<T>(arg)};
        }
    };

    namespace _impl {
        // replace the storage_info ID contained in a given storage with the new value
        template <unsigned Id, class DataStore>
        struct tmp_data_store {
            using type = DataStore;
        };

        template <unsigned Id, class Storage, class StorageInfo>
        struct tmp_data_store<Id, data_store<Storage, StorageInfo>> {
            using type = data_store<Storage,
                storage_info<Id,
                    typename StorageInfo::layout_t,
                    typename StorageInfo::halo_t,
                    typename StorageInfo::alignment_t>>;
        };

        template <unsigned Id, class DataStore>
        struct tmp_data_store<Id, std::vector<DataStore>> {
            using type = std::vector<typename tmp_data_store<Id, DataStore>::type>;
        };

        template <class Location>
        struct tmp_storage_info_id;
        template <int_t I, uint_t NColors>
        struct tmp_storage_info_id<location_type<I, NColors>> : std::integral_constant<unsigned, -NColors> {};

        template <uint_t>
        struct arg_tag;

    } // namespace _impl
    /** alias template that provides convenient tmp arg declaration.
     *
     *  Here we force tmp storages to share storage info type. To achieve this we substitute the storage info ID
     *  to one that is in the reserved range (close to max unsigned).
     *  TODO(anstaf): replace storage info IDs to tags to avoid having reserved range.
     */
    template <uint_t I, class DataStore, class Location = enumtype::default_location_type>
    using tmp_arg = plh<_impl::arg_tag<I>,
        typename _impl::tmp_data_store<_impl::tmp_storage_info_id<Location>::value, DataStore>::type,
        Location,
        true>;

    template <uint_t I, class DataStore, class LocationType = enumtype::default_location_type>
    using arg = plh<_impl::arg_tag<I>, DataStore, LocationType, false>;
} // namespace gridtools
