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

} // namespace gridtools::fn
