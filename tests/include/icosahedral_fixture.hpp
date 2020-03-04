/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>

#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#include "backend_select.hpp"
#include "float_type.hpp"
#include "grid_fixture.hpp"
#include "storage_select.hpp"

namespace gridtools {
    namespace icosahedral {
        template <size_t Halo, class Axis = axis<1>>
        struct computation_fixture : grid_fixture<Halo, Axis> {
            using grid_fixture<Halo, Axis>::grid_fixture;

            template <class Location, class T = float_type>
            auto builder() const {
                return storage::builder<storage_traits_t>                                //
                    .dimensions(this->d(0), this->d(1), this->k_size(), Location::value) //
                    .halos(Halo, Halo, 0, 0)                                             //
                    .template type<T>()                                                  //
                    .template id<Location::value>();
            }

            template <class Location, class T = float_type>
            auto make_storage() const {
                return builder<Location, T>().build();
            }

            template <class Location,
                class T = float_type,
                class U,
                std::enable_if_t<!std::is_convertible<U const &, T>::value, int> = 0>
            auto make_storage(U const &arg) const {
                return builder<Location, T>().initializer(arg).build();
            }

            template <class Location,
                class T = float_type,
                class U,
                std::enable_if_t<std::is_convertible<U const &, T>::value, int> = 0>
            auto make_storage(U const &arg) const {
                return builder<Location, T>().value(arg).build();
            }
        }; // namespace gridtools
    }      // namespace icosahedral
} // namespace gridtools
