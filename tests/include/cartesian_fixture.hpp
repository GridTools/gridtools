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

#include <gridtools/common/timer/timer.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#include "grid_fixture.hpp"
#include "storage_select.hpp"
#include "timer_select.hpp"

namespace gridtools {
    namespace cartesian {
        template <class DataType, class Backend, class Domain, size_t Halo = 0, class Axis = axis<1>>
        struct computation_fixture : grid_fixture<Domain, Halo, Axis> {
            template <class T = DataType>
            static auto builder() {
                return storage::builder<storage_traits_t>                                                            //
                    .dimensions(computation_fixture::d(0), computation_fixture::d(1), computation_fixture::k_size()) //
                    .halos(Halo, Halo, 0)                                                                            //
                    .template type<T>();
            }

            static Backend backend() { return {}; }

            using storage_type =
                decltype(storage::builder<storage_traits_t>.dimensions(0, 0, 0).template type<DataType>()());

            template <class T = DataType, class U, std::enable_if_t<!std::is_convertible<U const &, T>::value, int> = 0>
            static auto make_storage(U const &arg) {
                return builder<T>().initializer(arg).build();
            }

            template <class T = DataType, class U, std::enable_if_t<std::is_convertible<U const &, T>::value, int> = 0>
            static auto make_storage(U const &arg) {
                return builder<T>().value(arg).build();
            }

            template <class T = DataType>
            static auto make_storage() {
                return builder<T>().build();
            }

            template <class T = DataType, class U>
            static auto make_const_storage(U const &arg) {
                return make_storage<T const>(arg);
            }

            template <class Comp>
            static void benchmark(Comp &&comp) {
                size_t steps = Domain::steps();
                if (steps == 0)
                    return;
                comp();
                timer<timer_impl_t> timer = {"NoName"};
                for (size_t i = 0; i != steps; ++i) {
                    Domain::flush_cache();
                    timer.start();
                    comp();
                    timer.pause();
                }
                std::cout << timer.to_string() << std::endl;
            }
        };
    } // namespace cartesian
} // namespace gridtools
