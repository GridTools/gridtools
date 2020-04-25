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

#include <cstdlib>
#include <string>
#include <type_traits>
#include <typeinfo>

#include <gtest/gtest.h>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/timer/timer.hpp>
#include <gridtools/meta.hpp>
#include <gridtools/stencil/frontend/axis.hpp>
#include <gridtools/stencil/frontend/make_grid.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#include "timer_select.hpp"
#include "verifier.hpp"

#define GT_REGRESSION_TEST(name, env, backend)                                                             \
    template <class>                                                                                       \
    using name = ::testing::Test;                                                                          \
    using name##_types_t = meta::if_<env::is_enabled<backend>,                                             \
        ::testing::Types<env::apply<backend, double, ::gridtools::test_environment_impl_::cmdline_params>, \
            env::apply<backend, float, ::gridtools::test_environment_impl_::cmdline_params>,               \
            env::apply<backend, double, ::gridtools::test_environment_impl_::inlined_params<12, 33, 61>>,  \
            env::apply<backend, double, ::gridtools::test_environment_impl_::inlined_params<23, 11, 43>>>, \
        ::testing::Types<>>;                                                                               \
    TYPED_TEST_SUITE(name, name##_types_t, ::gridtools::test_environment_impl_::test_environment_names);   \
    TYPED_TEST(name, test)

namespace gridtools {
    namespace test_environment_impl_ {
        template <class T>
        std::true_type backend_supports_icosahedral(T);

        template <class T>
        std::true_type backend_supports_vertical_stencils(T);

        template <class T>
        std::is_same<decltype(backend_timer_impl(T())), timer_dummy> backend_skip_benchmark(T const &) {
            return {};
        }

        template <int... Is>
        struct inlined_params {
            static int d(size_t i) { return ((int[]){Is...})[i]; }
            static size_t steps() { return 0; }
            static bool needs_verification() { return true; }
            static std::string name() {
                std::string res = "_domain_size";
                for (int i : (int[]){Is...})
                    res += "_" + std::to_string(i);
                return res;
            }
        };

        template <class T>
        inline void flush_cache(T const &) {}

        void flush_cache(timer_omp const &);

        void add_time(std::string const &name, std::string const &backend, std::string const &float_type, double time);

        struct cmdline_params {
            static int d(size_t i);
            static size_t steps();
            static bool needs_verification();
            static std::string name() { return "_cmdline"; }
        };

        template <size_t Halo = 0,
            class Axis = stencil::axis<1>,
            class Pred = meta::always<std::true_type>,
            class = std::make_index_sequence<Axis::n_intervals>>
        struct test_environment;

        struct vertical_stencil {
            template <class Backend>
            using apply = decltype(backend_supports_vertical_stencils(Backend()));
        };

        struct icosahedral_stencil {
            template <class Backend>
            using apply = decltype(backend_supports_icosahedral(Backend()));
        };

        template <size_t Halo = 0, class Axis = stencil::axis<1>>
        using vertical_test_environment = test_environment<Halo, Axis, vertical_stencil>;

        template <size_t Halo = 0>
        using icosahedral_test_environment = test_environment<Halo, stencil::axis<1>, icosahedral_stencil>;

        template <size_t Halo, class Axis, class Pred, size_t... Is>
        struct test_environment<Halo, Axis, Pred, std::index_sequence<Is...>> {
            template <class Backend>
            using is_enabled = typename Pred::template apply<Backend>;

            template <class Backend, class FloatType, class ParamsSource>
            struct apply {
                using backend_t = Backend;
                using storage_traits_t = decltype(backend_storage_traits(Backend()));
                using timer_impl_t = decltype(backend_timer_impl(Backend()));
                using float_t = FloatType;

                static auto d(size_t i) { return ParamsSource::d(i) + (i < 2 ? Halo * 2 : 0); }

                static auto k_size() { return make_grid().k_size(typename Axis::full_interval()); }

                static auto make_grid() {
                    auto halo_desc = [](auto d) { return halo_descriptor(Halo, Halo, Halo, d - Halo - 1, d); };
                    return stencil::make_grid(halo_desc(d(0)), halo_desc(d(1)), Axis(d(2 + Is)...));
                }

                template <class Expected, class Actual, class EqualTo = default_equal_to<typename Actual::element_type>>
                static void verify(Expected const &expected, Actual const &actual, EqualTo equal_to = {}) {
                    if (!ParamsSource::needs_verification())
                        return;
                    std::array<std::array<size_t, 2>, Actual::element_type::ndims> halos = {
                        {{Halo, Halo}, {Halo, Halo}}};
                    EXPECT_TRUE(verify_data_store(expected, actual, halos, equal_to));
                }

                template <class T = FloatType>
                static auto builder() {
                    return storage::builder<storage_traits_t> //
                        .dimensions(d(0), d(1), k_size())     //
                        .halos(Halo, Halo, 0)                 //
                        .template type<T>();
                }

                static Backend backend() { return {}; }

                using storage_type =
                    decltype(storage::builder<storage_traits_t>.dimensions(0, 0, 0).template type<FloatType>()());

                template <class T = FloatType,
                    class U,
                    std::enable_if_t<!std::is_convertible<U const &, T>::value, int> = 0>
                static auto make_storage(U const &arg) {
                    return builder<T>().initializer(arg).build();
                }

                template <class T = FloatType,
                    class U,
                    std::enable_if_t<std::is_convertible<U const &, T>::value, int> = 0>
                static auto make_storage(U const &arg) {
                    return builder<T>().value(arg).build();
                }

                template <class T = FloatType>
                static auto make_storage() {
                    return builder<T>().build();
                }

                template <class T = FloatType, class U>
                static auto make_const_storage(U const &arg) {
                    return make_storage<T const>(arg);
                }

                template <class T = FloatType, class Location>
                static auto icosahedral_builder(Location) {
                    return storage::builder<storage_traits_t>              //
                        .dimensions(d(0), d(1), k_size(), Location::value) //
                        .halos(Halo, Halo, 0, 0)                           //
                        .template type<T>()                                //
                        .template id<Location::value>();
                }

                template <class T = FloatType, class Location>
                static auto icosahedral_make_storage(Location loc) {
                    return icosahedral_builder<T>(loc).build();
                }

                template <class T = FloatType,
                    class Location,
                    class U,
                    std::enable_if_t<!std::is_convertible<U const &, T>::value, int> = 0>
                static auto icosahedral_make_storage(Location loc, U const &arg) {
                    return icosahedral_builder<T>(loc).initializer(arg).build();
                }

                template <class T = FloatType,
                    class Location,
                    class U,
                    std::enable_if_t<std::is_convertible<U const &, T>::value, int> = 0>
                static auto icosahedral_make_storage(Location loc, U const &arg) {
                    return icosahedral_builder<T>(loc).value(arg).build();
                }

                template <class Comp>
                static void benchmark(std::string const &name, Comp &&comp) {
                    size_t steps = ParamsSource::steps();
                    if (steps == 0 || backend_skip_benchmark(Backend()))
                        return;
                    comp();
                    timer_impl_t timer;
                    for (size_t i = 0; i != steps; ++i) {
                        flush_cache(timer);
                        timer.start_impl();
                        comp();
                        auto time = timer.pause_impl();
                        add_time(name, backend_name(Backend()), float_type_name(), time);
                    }
                }

                static auto test_name() {
                    return std::string() + backend_name(Backend()) + "_" + float_type_name() + ParamsSource::name();
                }

              private:
                static auto float_type_name() {
                    return std::is_same<FloatType, float>::value
                               ? "float"
                               : std::is_same<FloatType, double>::value ? "double" : typeid(FloatType).name();
                }
            };
        };

        struct test_environment_names {
            template <class T>
            static auto GetName(int) {
                return T::test_name();
            }
        };
    } // namespace test_environment_impl_

    using test_environment_impl_::icosahedral_test_environment;
    using test_environment_impl_::inlined_params;
    using test_environment_impl_::test_environment;
    using test_environment_impl_::vertical_test_environment;
} // namespace gridtools
