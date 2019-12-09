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

#include "../storage/builder.hpp"
#include "../storage/sid.hpp"
#include "backend_select.hpp"
#include "grid_fixture.hpp"

namespace gridtools {
    namespace cartesian {
        template <size_t Halo = 0, class Axis = axis<1>>
        struct computation_fixture : grid_fixture<Halo, Axis> {
            using grid_fixture<Halo, Axis>::grid_fixture;

            template <class T = float_type>
            auto builder() const {
                return storage::builder<storage_traits_t>               //
                    .dimensions(this->d(0), this->d(1), this->k_size()) //
                    .halos(Halo, Halo, 0)                               //
                    .template type<T>();
            }

            using storage_type =
                decltype(storage::builder<storage_traits_t>.dimensions(0, 0, 0).template type<float_type>()());

            template <class T = float_type,
                class U,
                std::enable_if_t<!std::is_convertible<U const &, T>::value, int> = 0>
            auto make_storage(U const &arg) const {
                return builder<T>().initializer(arg).build();
            }

            template <class T = float_type,
                class U,
                std::enable_if_t<std::is_convertible<U const &, T>::value, int> = 0>
            auto make_storage(U const &arg) const {
                return builder<T>().value(arg).build();
            }

            template <class T = float_type>
            auto make_storage() const {
                return builder<T>().build();
            }

            template <class T = float_type, class U>
            auto make_const_storage(U const &arg) const {
                return make_storage<T const>(arg);
            }
        };
    } // namespace cartesian
} // namespace gridtools
