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

#include <memory>
#include <string>
#include <utility>

#include "../common/defs.hpp"
#include "../common/permute_to.hpp"
#include "../meta/type_traits.hpp"
#include "accessor_intent.hpp"
#include "arg.hpp"
#include "extent.hpp"

namespace gridtools {

    namespace _impl {
        namespace computation_detail {
            template <class Obj>
            struct run_f {
                Obj &m_obj;

                template <class... Args>
                void operator()(Args &&... args) const {
                    m_obj.run(wstd::forward<Args>(args)...);
                }
            };

            template <typename Arg>
            struct iface_arg {
                virtual ~iface_arg() = default;
                virtual rt_extent get_arg_extent(Arg) const = 0;
                virtual intent get_arg_intent(Arg) const = 0;
            };

            template <class T, class Arg>
            struct impl_arg : virtual iface_arg<Arg> {
                rt_extent get_arg_extent(Arg) const override {
                    return static_cast<const T *>(this)->m_obj.get_arg_extent(Arg());
                }
                intent get_arg_intent(Arg) const override {
                    return static_cast<const T *>(this)->m_obj.get_arg_intent(Arg());
                }
            };
        } // namespace computation_detail
    }     // namespace _impl

    /**
     * Type erasure for computations (the objects that are produced by make_computation)
     * Note that it is move only (no copy constructor)
     *
     * @tparam Args placeholders that should be passed to run as corespondent arg_storage_pairs
     */
    template <class... Args>
    class computation {
        GT_STATIC_ASSERT(conjunction<is_plh<Args>...>::value, "template parameters should be args");

        using arg_storage_pair_crefs_t = std::tuple<arg_storage_pair<Args, typename Args::data_store_t> const &...>;

        struct iface : virtual _impl::computation_detail::iface_arg<Args>... {
            virtual ~iface() = default;
            virtual void run(arg_storage_pair_crefs_t const &) = 0;
            virtual std::string print_meter() const = 0;
            virtual double get_time() const = 0;
            virtual size_t get_count() const = 0;
            virtual void reset_meter() = 0;
        };

        template <class Obj>
        struct impl : iface, _impl::computation_detail::impl_arg<impl<Obj>, Args>... {
            Obj m_obj;

            impl(Obj &&obj) : m_obj{wstd::move(obj)} {}

            void run(arg_storage_pair_crefs_t const &args) override {
                tuple_util::apply(_impl::computation_detail::run_f<Obj>{m_obj}, args);
            }
            std::string print_meter() const override { return m_obj.print_meter(); }
            double get_time() const override { return m_obj.get_time(); }
            size_t get_count() const override { return m_obj.get_count(); }
            void reset_meter() override { m_obj.reset_meter(); }
        };

        std::unique_ptr<iface> m_impl;

      public:
        computation() = default;

        template <class Obj>
        computation(Obj obj) : m_impl(new impl<Obj>{wstd::move(obj)}) {
            GT_STATIC_ASSERT((!std::is_same<typename std::decay<Obj>::type, computation>::value),
                GT_INTERNAL_ERROR_MSG("computation move ctor got shadowed"));
            // TODO(anstaf): Check that Obj satisfies computation concept here.
        }

        explicit operator bool() const { return !!m_impl; }

        template <class... SomeArgs, class... SomeDataStores>
        typename std::enable_if<sizeof...(SomeArgs) == sizeof...(Args)>::type run(
            arg_storage_pair<SomeArgs, SomeDataStores> const &... args) {
            m_impl->run(permute_to<arg_storage_pair_crefs_t>(std::make_tuple(std::cref(args)...)));
        }

        std::string print_meter() const { return m_impl->print_meter(); }

        double get_time() const { return m_impl->get_time(); }

        size_t get_count() const { return m_impl->get_count(); }

        void reset_meter() { m_impl->reset_meter(); }

        template <class Arg>
        std::enable_if_t<meta::st_contains<meta::list<Args...>, Arg>::value, rt_extent> get_arg_extent(Arg) const {
            return static_cast<_impl::computation_detail::iface_arg<Arg> const &>(*m_impl).get_arg_extent(Arg());
        }

        template <class Arg>
        std::enable_if_t<meta::st_contains<meta::list<Args...>, Arg>::value, intent> get_arg_intent(Arg) const {
            return static_cast<_impl::computation_detail::iface_arg<Arg> const &>(*m_impl).get_arg_intent(Arg());
        }
    };

} // namespace gridtools
