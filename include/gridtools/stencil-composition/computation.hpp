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

#include <boost/fusion/include/invoke.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/make_vector.hpp>

#include "arg.hpp"
#include "../common/defs.hpp"
#include "../common/permute_to.hpp"
#include "../common/generic_metafunctions/meta.hpp"

namespace gridtools {

    namespace _impl {
        namespace computation_detail {

            template < class ReturnType, class Obj >
            struct run_f {
                Obj &m_obj;

                template < class... Args >
                ReturnType operator()(Args &&... args) const {
                    return m_obj.run(std::forward< Args >(args)...);
                }
                using result_type = ReturnType;
            };
            template < class Obj >
            struct run_f< void, Obj > {
                Obj &m_obj;

                template < class... Args >
                void operator()(Args &&... args) const {
                    m_obj.run(std::forward< Args >(args)...);
                }
                using result_type = void;
            };
            template < class ReturnType, class Obj, class Args >
            ReturnType invoke_run(Obj &obj, Args const &args) {
                return boost::fusion::invoke(run_f< ReturnType, Obj >{obj}, args);
            }
        }
    }

    /**
     * Type erasure for computations (the objects that are produced by make_computation)
     * Note that it is move only (no copy costructor)
     *
     * @tparam ReturnType what is returned by run method
     * @tparam Args placeholders that should be passed to run as corespondent arg_storage_pairs
     */
    template < class ReturnType, class... Args >
    class computation {
        GRIDTOOLS_STATIC_ASSERT(meta::conjunction< is_arg< Args >... >::value, "template parameters should be args");

        using arg_storage_pair_crefs_t =
            boost::fusion::vector< arg_storage_pair< Args, typename Args::data_store_t > const &... >;

        struct iface {
            virtual ~iface() = default;
            virtual ReturnType run(arg_storage_pair_crefs_t const &) = 0;
            virtual void sync_all() = 0;
            virtual std::string print_meter() const = 0;
            virtual double get_meter() const = 0;
            virtual void reset_meter() = 0;
        };

        template < class Obj >
        struct impl : iface {
            Obj m_obj;

            impl(Obj &&obj) : m_obj{std::move(obj)} {}

            ReturnType run(arg_storage_pair_crefs_t const &args) override {
                return _impl::computation_detail::invoke_run< ReturnType >(m_obj, args);
            }
            void sync_all() override { m_obj.sync_all(); }
            std::string print_meter() const override { return m_obj.print_meter(); }
            double get_meter() const override { return m_obj.get_meter(); }
            void reset_meter() override { return m_obj.reset_meter(); }
        };

        std::unique_ptr< iface > m_impl;

      public:
        computation() = default;

        template < class Obj >
        computation(Obj obj)
            : m_impl(new impl< Obj >{std::move(obj)}) {
            GRIDTOOLS_STATIC_ASSERT((!std::is_same< typename std::decay< Obj >::type, computation >::value),
                GT_INTERNAL_ERROR_MSG("computation move ctor got shadowed"));
            // TODO(anstaf): Check that Obj satisfies computation concept here.
        }

        explicit operator bool() const { return !!m_impl; }

        template < class... SomeArgs, class... SomeDataStores >
        typename std::enable_if< sizeof...(SomeArgs) == sizeof...(Args), ReturnType >::type run(
            arg_storage_pair< SomeArgs, SomeDataStores > const &... args) {
            return m_impl->run(permute_to< arg_storage_pair_crefs_t >(boost::fusion::make_vector(std::cref(args)...)));
        }

        void sync_all() { m_impl->sync_all(); }

        std::string print_meter() const { return m_impl->print_meter(); }

        double get_meter() const { return m_impl->get_meter(); }

        void reset_meter() { m_impl->reset_meter(); }
    };

} // namespace gridtools
