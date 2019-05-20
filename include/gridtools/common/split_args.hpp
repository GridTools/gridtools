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
#include <type_traits>
#include <utility>

#include "../meta/filter.hpp"
#include "../meta/first.hpp"
#include "../meta/macros.hpp"
#include "../meta/make_indices.hpp"
#include "../meta/not.hpp"
#include "../meta/second.hpp"
#include "../meta/transform.hpp"
#include "../meta/type_traits.hpp"
#include "../meta/zip.hpp"
#include "defs.hpp"
#include "generic_metafunctions/utility.hpp"

namespace gridtools {
    namespace _impl {
        namespace _split_args {
            template <template <class...> class Pred>
            struct apply_to_first {
                template <class L>
                using apply = Pred<meta::first<L>>;
            };

            template <template <class...> class Pred>
            struct apply_to_decayed {
                template <class T>
                using apply = Pred<decay_t<T>>;
            };

            template <template <class...> class Pred, class Args>
            using make_filtered_indicies = meta::transform<meta::second,
                meta::filter<apply_to_first<Pred>::template apply, meta::zip<Args, meta::make_indices_for<Args>>>>;

            template <class Args, template <class...> class L, class... Is>
            auto get_part_helper(Args &&args, L<Is...> *) {
                return std::forward_as_tuple(std::get<Is::value>(wstd::forward<Args>(args))...);
            }

            template <template <class...> class Pred, class Args>
            auto get_part(Args &&args) {
                return get_part_helper(wstd::forward<Args>(args), (make_filtered_indicies<Pred, Args> *)(nullptr));
            }

            template <template <class...> class Pred, class Args>
            auto raw_split_args_tuple(Args &&args) {
                return std::make_pair(get_part<Pred>(wstd::forward<Args>(args)),
                    get_part<meta::not_<Pred>::template apply>(wstd::forward<Args>(args)));
            }

            template <template <class...> class Pred, class Args>
            auto split_args_tuple(Args &&args) {
                return raw_split_args_tuple<apply_to_decayed<Pred>::template apply>(wstd::forward<Args>(args));
            }
        } // namespace _split_args
    }     // namespace _impl

    /// Variations that take a tuple instead of parameter pack
    using _impl::_split_args::raw_split_args_tuple;
    using _impl::_split_args::split_args_tuple;

    /**
     *  Split the args into two groups according to the given compile time predicate on the argument type
     *  Argument types are taken raw into predicate . With references and const modifiers.
     *
     * @return std::pair of two std::tuples. First tuple is from the types that satisfies predicate Pred
     */
    template <template <class...> class Pred, class... Args>
    auto raw_split_args(Args &&... args) {
        return raw_split_args_tuple<Pred>(std::forward_as_tuple(wstd::forward<Args>(args)...));
    }

    /// A handy variation of raw_split_args that applies predicate on decayed argument types.
    template <template <class...> class Pred, class... Args>
    auto split_args(Args &&... args) {
        return split_args_tuple<Pred>(std::forward_as_tuple(wstd::forward<Args>(args)...));
    }
} // namespace gridtools
