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

namespace gridtools::fn {

    template <class Backend, class MakeIterator, class Domain, class Args = std::tuple<>, class Stages = meta::list<>>
    class stencil_executor {
        Domain m_domain;
        Args m_args;

      public:
        stencil_executor(Backend, MakeIterator, Domain const &domain) : m_domain(domain) {}
        stencil_executor(Domain &&domain, Args &&args) : m_domain(std::move(domain)), m_args(std::move(args)) {}
        stencil_executor(stencil_executor const &) = delete;
        stencil_executor(stencil_executor &&) = default;
        ~stencil_executor() { run_stencils(Backend(), Stages(), m_domain, std::move(m_args)); }

        template <class Arg>
        auto arg(Arg &&arg) && {
            auto args = tuple_util::push_back(std::move(m_args), std::forward<Arg>(arg));
            return stencil_executor<Backend, MakeIterator, Domain, decltype(args), Stages>{
                std::move(m_domain), std::move(args)};
        }
        template <class Out, class Stencil, class... Ins>
        auto assign(Out, Stencil, Ins...) && {
            using stages_t = meta::push_back<Stages, stencil_stage<Stencil, MakeIterator, Out::value, Ins::value...>>;
            return stencil_executor<Backend, MakeIterator, Domain, Args, stages_t>{
                std::move(m_domain), std::move(m_args)};
        }
    };

    template <class Backend,
        class MakeIterator,
        class Domain,
        class Vertical,
        class Args = std::tuple<>,
        class Stages = meta::list<>,
        class Seeds = std::tuple<>>
    class vertical_executor {
        Domain m_domain;
        Args m_args;
        Seeds m_seeds;

      public:
        vertical_executor(Backend, MakeIterator, Domain const &domain, Vertical) : m_domain(domain) {}
        vertical_executor(Domain &&domain, Args &&args, Seeds &&seeds)
            : m_domain(std::move(domain)), m_args(std::move(args)), m_seeds(std::move(seeds)) {}
        vertical_executor(vertical_executor const &) = delete;
        vertical_executor(vertical_executor &&) = default;
        ~vertical_executor() {
            run_vertical(Backend(), Stages(), m_domain, Vertical(), std::move(m_args), std::move(m_seeds));
        }

        template <class Arg>
        auto arg(Arg &&arg) && {
            auto args = tuple_util::push_back(std::move(m_args), std::forward<Arg>(arg));
            return vertical_executor<Backend, MakeIterator, Domain, Vertical, decltype(args), Stages, Seeds>{
                std::move(m_domain), std::move(args), std::move(m_seeds)};
        }
        template <class Out, class Stencil, class Seed, class... Ins>
        auto assign(Out, Stencil, Seed &&seed, Ins...) && {
            using stages_t =
                meta::push_back<Stages, column_stage<Vertical, Stencil, MakeIterator, Out::value, Ins::value...>>;
            auto seeds = tuple_util::push_back(std::move(m_seeds), std::forward<Seed>(seed));
            return vertical_executor<Backend, MakeIterator, Domain, Vertical, Args, stages_t, decltype(seeds)>{
                std::move(m_domain), std::move(m_args), std::move(seeds)};
        }
    };

} // namespace gridtools::fn
