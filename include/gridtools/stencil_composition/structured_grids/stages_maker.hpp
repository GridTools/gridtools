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

#include "../../common/defs.hpp"
#include "../../meta.hpp"
#include "../bind_functor_with_interval.hpp"
#include "../compute_extents_metafunctions.hpp"
#include "../mss.hpp"
#include "esf.hpp"
#include "stage.hpp"

namespace gridtools {

    namespace stages_maker_impl_ {
        template <class Functor, class Esf, class ExtentMap>
        struct stages_from_functor {
            using extent_t = get_esf_extent<Esf, ExtentMap>;
            using type = meta::list<stage<Functor, extent_t, Esf>>;
        };
        template <class Esf, class ExtentMap>
        struct stages_from_functor<void, Esf, ExtentMap> {
            using type = meta::list<>;
        };

        template <class Index, class ExtentMap>
        struct stages_from_esf_f {
            template <class Esf>
            using apply = typename stages_from_functor<bind_functor_with_interval<typename Esf::esf_function_t, Index>,
                Esf,
                ExtentMap>::type;
        };

        template <class Esfs, class Index, class ExtentMap>
        using stages_from_esfs =
            meta::flatten<meta::transform<stages_from_esf_f<Index, ExtentMap>::template apply, Esfs>>;
    } // namespace stages_maker_impl_

    /**
     *   Transforms `mss_descriptor` into a sequence of stages.
     *
     * @tparam Descriptor -   mss descriptor
     * @tparam ExtentMap -    a compile time map that maps placeholders to computed extents.
     *                        `stages_maker` uses ExtentMap parameter in an opaque way -- it just delegates it to
     *                        `get_extent_for` when it is needed.
     *
     *   This metafunction returns another metafunction (i.e. has nested `apply` metafunction) that accepts
     *   a single argument that has to be a level_index and returns the stages (classes that model Stage concept)
     *
     *   Note that the collection of stages is calculated for the provided level_index. It can happen that same mss
     *   produces different stages for the different level indices because of interval overloads. If for two level
     *   indices all elementary functors should be called with the same correspondent intervals, the returned stages
     *   will be exactly the same.
     *
     *   TODO(anstaf): unit test!!!
     */
    template <class Mss, class ExtentMap>
    struct stages_maker {
        template <class LevelIndex>
        using apply = stages_maker_impl_::stages_from_esfs<typename Mss::esf_sequence_t, LevelIndex, ExtentMap>;
    };
} // namespace gridtools
