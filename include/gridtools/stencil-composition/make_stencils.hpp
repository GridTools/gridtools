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
    } // namespace _impl

    /*!
       \brief Function to create a Multistage Stencil that can then be executed
       \param esf{i}  i-th Elementary Stencil Function created with ::gridtools::make_stage or a list specified as
       independent ESF created with ::gridtools::make_independent

       Use this function to create a multi-stage stencil computation
     */
    template <typename ExecutionEngine, typename... MssParameters>
    mss_descriptor<ExecutionEngine,
        GT_META_CALL(extract_mss_esfs, (MssParameters...)),
        typename extract_mss_caches<MssParameters...>::type>
    make_multistage(ExecutionEngine, MssParameters...) {
        GT_STATIC_ASSERT((is_execution_engine<ExecutionEngine>::value),
            "The first argument passed to make_multistage must be the execution engine (e.g. execute::forward(), "
            "execute::backward(), execute::parallel())");
        GT_STATIC_ASSERT(conjunction<is_mss_parameter<MssParameters>...>::value,
            "wrong set of mss parameters passed to make_multistage construct.\n"
            "Check that arguments passed are either :\n"
            " * caches from define_caches(...) construct or\n"
            " * esf descriptors from make_stage(...) or make_independent(...)");
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
