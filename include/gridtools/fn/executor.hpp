/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <tuple>

#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "./run.hpp"
#include "./scan.hpp"
#include "./stencil_stage.hpp"
#include "gridtools/meta/debug.hpp"

namespace gridtools::fn {
    namespace executor_impl_ {
        template <class MakeSpec,
            class RunSpecs,
            class Domain,
            class MakeIterator,
            class Args = std::tuple<>,
            class Specs = std::tuple<>>
        struct executor {
            Domain m_domain;
            MakeIterator m_make_iterator;
            Args m_args = {};
            Specs m_specs = {};
            bool m_active = true;

            executor(executor const &) = delete;
            ~executor() {
                if (m_active)
                    RunSpecs()(m_domain, m_make_iterator, std::move(m_args), std::move(m_specs));
            }

            template <class Arg>
            auto arg(Arg &&arg) && {
                assert(m_active);
                auto args = tuple_util::push_back(std::move(m_args), std::forward<Arg>(arg));
                m_active = false;
                return executor<MakeSpec, RunSpecs, Domain, MakeIterator, decltype(args), Specs>{
                    std::move(m_domain), std::move(m_make_iterator), std::move(args), std::move(m_specs)};
            }

            template <class... SpecArgs>
            auto assign(SpecArgs &&...args) && {
                assert(m_active);
                auto specs = tuple_util::deep_copy(
                    tuple_util::push_back(std::move(m_specs), MakeSpec()(std::forward<SpecArgs>(args)...)));
                m_active = false;
                return executor<MakeSpec, RunSpecs, Domain, MakeIterator, Args, decltype(specs)>{
                    std::move(m_domain), std::move(m_make_iterator), std::move(m_args), std::move(specs)};
            }
        };

        struct make_stencil_spec_f {
            template <class Out, class Stencil, class... Ins>
            constexpr auto operator()(Out, Stencil, Ins...) const {
                return stencil_stage<Stencil, Out::value, Ins::value...>();
            }
        };

        template <class Backend>
        struct run_stencil_specs_f {
            template <class Domain, class MakeIterator, class Args, class Specs>
            constexpr auto operator()(Domain const &domain, MakeIterator const &make_iterator, Args args, Specs) const {
                using stages_t = meta::rename<meta::list, Specs>;
                run_stencils(Backend(), stages_t(), make_iterator, domain, std::move(args));
            }
        };

        template <class Backend, class MakeIterator, class Domain>
        using stencil_executor = executor<make_stencil_spec_f, run_stencil_specs_f<Backend>, Domain, MakeIterator>;

        template <class Vertical>
        struct make_vertical_spec_f {
            template <class Out, class ScanOrFold, class Seed, class... Ins>
            constexpr auto operator()(Out, ScanOrFold, Seed &&seed, Ins...) const {
                return std::tuple(
                    column_stage<Vertical, ScanOrFold, Out::value, Ins::value...>(), std::forward<Seed>(seed));
            }
        };

        template <class Backend, class Vertical>
        struct run_vertical_specs_f {
            template <class Domain, class MakeIterator, class Args, class Specs>
            constexpr auto operator()(
                Domain const &domain, MakeIterator const &make_iterator, Args args, Specs &&specs) const {
                using stages_t = meta::transform<meta::first,
                    meta::transform<std::remove_reference_t, meta::rename<meta::list, Specs>>>;
                auto seeds = tuple_util::transform([](auto &&t) { return tuple_util::get<1>(t); }, std::move(specs));
                run_vertical(
                    Backend(), stages_t(), make_iterator, domain, Vertical(), std::move(args), std::move(seeds));
            }
        };

        template <class Backend, class MakeIterator, class Vertical, class Domain>
        using vertical_executor =
            executor<make_vertical_spec_f<Vertical>, run_vertical_specs_f<Backend, Vertical>, Domain, MakeIterator>;
    } // namespace executor_impl_

    using executor_impl_::stencil_executor;
    using executor_impl_::vertical_executor;
} // namespace gridtools::fn
