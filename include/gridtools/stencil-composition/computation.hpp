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
                    m_obj.run(std::forward<Args>(args)...);
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
            virtual void sync_bound_data_stores() = 0;
            virtual std::string print_meter() const = 0;
            virtual double get_time() const = 0;
            virtual size_t get_count() const = 0;
            virtual void reset_meter() = 0;
        };

        template <class Obj>
        struct impl : iface, _impl::computation_detail::impl_arg<impl<Obj>, Args>... {
            Obj m_obj;

            impl(Obj &&obj) : m_obj{std::move(obj)} {}

            void run(arg_storage_pair_crefs_t const &args) override {
                tuple_util::apply(_impl::computation_detail::run_f<Obj>{m_obj}, args);
            }
            void sync_bound_data_stores() override { m_obj.sync_bound_data_stores(); }
            std::string print_meter() const override { return m_obj.print_meter(); }
            double get_time() const override { return m_obj.get_time(); }
            size_t get_count() const override { return m_obj.get_count(); }
            void reset_meter() override { m_obj.reset_meter(); }
        };

        std::unique_ptr<iface> m_impl;

      public:
        computation() = default;

        template <class Obj>
        computation(Obj obj) : m_impl(new impl<Obj>{std::move(obj)}) {
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

        void sync_bound_data_stores() { m_impl->sync_bound_data_stores(); }

        std::string print_meter() const { return m_impl->print_meter(); }

        double get_time() const { return m_impl->get_time(); }

        size_t get_count() const { return m_impl->get_count(); }

        void reset_meter() { m_impl->reset_meter(); }

        template <class Arg>
        enable_if_t<meta::st_contains<meta::list<Args...>, Arg>::value, rt_extent> get_arg_extent(Arg) const {
            return static_cast<_impl::computation_detail::iface_arg<Arg> const &>(*m_impl).get_arg_extent(Arg());
        }

        template <class Arg>
        enable_if_t<meta::st_contains<meta::list<Args...>, Arg>::value, intent> get_arg_intent(Arg) const {
            return static_cast<_impl::computation_detail::iface_arg<Arg> const &>(*m_impl).get_arg_intent(Arg());
        }
    };

} // namespace gridtools
