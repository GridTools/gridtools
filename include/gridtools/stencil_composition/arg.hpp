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

    template <class Tag, class DataStore, class Location>
    struct plh;

    template <class>
    struct is_plh : std::false_type {};

    template <class Tag, class DataStore, class Location>
    struct is_plh<plh<Tag, DataStore, Location>> : std::true_type {};

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
     */
    template <class Tag, class DataStore, class Location>
    struct plh {
        GT_STATIC_ASSERT(
            is_location_type<Location>::value, "The third template argument of a placeholder must be a location_type");
        using data_store_t = DataStore;
        using location_t = Location;
        using tag_t = Tag;
        using is_expandable_t = meta::is_instantiation_of<std::vector, DataStore>;

        template <class T>
        arg_storage_pair<plh, T> operator=(T &&arg) const {
            return {std::forward<T>(arg)};
        }
    };

    template <class Tag, class Data, class Location>
    struct tmp_plh {
        GT_STATIC_ASSERT(
            is_location_type<Location>::value, "The third template argument of a placeholder must be a location_type");
        using location_t = Location;
        using tag_t = Tag;
        using data_t = Data;
        using is_expandable_t = meta::is_instantiation_of<std::vector, Data>;
    };

    template <class T>
    using is_tmp_arg = meta::is_instantiation_of<tmp_plh, T>;

    template <class Tag, class T, class Location>
    struct is_plh<tmp_plh<Tag, T, Location>> : std::true_type {};

    namespace _impl {
        template <uint_t>
        struct arg_tag;

        template <class DataStore, class = void>
        struct tmp_data_type {
            using type = DataStore;
        };

        template <class DataStore>
        struct tmp_data_type<DataStore, void_t<typename DataStore::data_t>> {
            using type = typename DataStore::data_t;
        };

        template <class DataStore>
        struct tmp_data_type<std::vector<DataStore>> {
            using type = std::vector<typename tmp_data_type<DataStore>::type>;
        };
    } // namespace _impl

    template <uint_t I, class DataStore, class Location = enumtype::default_location_type>
    using tmp_arg = tmp_plh<_impl::arg_tag<I>, typename _impl::tmp_data_type<DataStore>::type, Location>;

    template <uint_t I, class DataStore, class LocationType = enumtype::default_location_type>
    using arg = plh<_impl::arg_tag<I>, DataStore, LocationType>;

} // namespace gridtools
