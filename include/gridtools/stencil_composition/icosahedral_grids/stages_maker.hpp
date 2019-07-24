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
#include "./esf.hpp"
#include "./stage.hpp"

namespace gridtools {

    namespace stages_maker_impl_ {

        template <class Index>
        struct bind_functor_with_interval_f {
            template <class Functor>
            using apply = bind_functor_with_interval<Functor, Index>;
        };

        template <class Esf>
        struct esf_functor_f {
            template <class Color>
            using apply = typename Esf::template esf_function<Color::value>;
        };

        template <class Esf>
        using get_functors = meta::transform<esf_functor_f<Esf>::template apply,
            meta::make_indices_c<Esf::location_type::n_colors::value>>;

        template <class Functors, class Esf, class ExtentMap, class = void>
        struct stages_from_functors {
            using extent_t = get_esf_extent<Esf, ExtentMap>;
            using type = meta::list<stage<Functors, extent_t, Esf>>;
        };
        template <class Functors, class Esf, class ExtentMap>
        struct stages_from_functors<Functors,
            Esf,
            ExtentMap,
            std::enable_if_t<meta::all_of<std::is_void, Functors>::value>> {
            using type = meta::list<>;
        };

        template <class Index, class ExtentMap>
        struct stages_from_esf_f {
            template <class Esf>
            using apply = typename stages_from_functors<
                meta::transform<bind_functor_with_interval_f<Index>::template apply, get_functors<Esf>>,
                Esf,
                ExtentMap>::type;
        };
    } // namespace stages_maker_impl_

    /**
     *   Transforms mss_descriptor into a sequence of stages.
     *
     * @tparam MssDescriptor -   mss descriptor
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
    template <class Mss, class ExtentMap = get_extent_map_from_mss<Mss>>
    struct stages_maker {
        using esfs_t = typename Mss::esf_sequence_t;
        template <class LevelIndex>
        using apply = meta::flatten<
            meta::transform<stages_maker_impl_::stages_from_esf_f<LevelIndex, ExtentMap>::template apply, esfs_t>>;
    };
} // namespace gridtools
