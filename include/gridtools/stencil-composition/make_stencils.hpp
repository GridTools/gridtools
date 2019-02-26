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

#include <tuple>

#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/std_tuple.hpp>

#include "../common/defs.hpp"
#include "../meta/flatten.hpp"
#include "../meta/list.hpp"
#include "../meta/macros.hpp"
#include "../meta/type_traits.hpp"
#include "independent_esf.hpp"
#include "mss.hpp"
#include "mss_metafunctions.hpp"

namespace gridtools {
    namespace _impl {
        template <class Esf>
        struct tuple_from_esf {
            using type = std::tuple<Esf>;
        };
        template <class Esfs>
        struct tuple_from_esf<independent_esf<Esfs>> {
            using type = Esfs;
        };

        template <class... Esfs>
        GT_META_DEFINE_ALIAS(
            tuple_from_esfs, meta::flatten, (meta::list<std::tuple<>, typename tuple_from_esf<Esfs>::type...>));

        template <typename ExecutionEngine, typename... MssParameters>
        struct check_make_multistage_args : std::true_type {
            GT_STATIC_ASSERT((is_execution_engine<ExecutionEngine>::value),
                "The first argument passed to make_multistage must be the execution engine (e.g. execute::forward(), "
                "execute::backward(), execute::parallel()");
            GT_STATIC_ASSERT(conjunction<is_mss_parameter<MssParameters>...>::value,
                "wrong set of mss parameters passed to make_multistage construct.\n"
                "Check that arguments passed are either :\n"
                " * caches from define_caches(...) construct or\n"
                " * esf descriptors from make_stage(...) or make_independent(...)");
        };
    } // namespace _impl

    /*!
       \brief Function to create a Multistage Stencil that can then be executed
       \param esf{i}  i-th Elementary Stencil Function created with ::gridtools::make_stage or a list specified as
       independent ESF created with ::gridtools::make_independent

       Use this function to create a multi-stage stencil computation
     */
    template <typename ExecutionEngine,
        typename... MssParameters,
        // Check argument types before mss_descriptor is instantiated to get nicer error messages
        bool ArgsOk = _impl::check_make_multistage_args<ExecutionEngine, MssParameters...>::value>
    mss_descriptor<ExecutionEngine,
        GT_META_CALL(extract_mss_esfs, (MssParameters...)),
        typename extract_mss_caches<MssParameters...>::type>
    make_multistage(ExecutionEngine, MssParameters...) {
        return {};
    }

    /*!
       \brief Function to create a list of independent Elementary Stencil Functions

       \param esf{i}  (must be i>=2) The max{i} Elementary Stencil Functions in the argument list will be treated as
       independent

       Function to create a list of independent Elementary Stencil Functions. This is used to let the library compute
       tight bounds on blocks to be used by backends

       _impl::tuple_from_esfs is used here to flatten the Esfs within independent_esf. It ensures that nested
       make_independent calls produces a single independent_esf
       for example:
       make_independent(make_independent(f1, f2), f3) will produce independent_esf<tuple<f1, f2, f3>>
     */
    template <class Esf1, class Esf2, class... Esfs>
    independent_esf<GT_META_CALL(_impl::tuple_from_esfs, (Esf1, Esf2, Esfs...))> make_independent(Esf1, Esf2, Esfs...) {
        return {};
    }

} // namespace gridtools
