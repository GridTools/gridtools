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

#include "../../../common/defs.hpp"
#include "../../../common/hymap.hpp"
#include "../../../meta.hpp"
#include "../../../storage/sid.hpp"
#include "../../arg.hpp"
#include "../../dim.hpp"
#include "../../extent.hpp"
#include "../../extract_placeholders.hpp"
#include "../../positional.hpp"
#include "../../sid/composite.hpp"
#include "../../sid/concept.hpp"

namespace gridtools {
    namespace x86 {
        namespace local_domain_impl_ {
            namespace lazy {
                template <class Arg>
                struct get_storage {
                    using type = typename Arg::data_store_t;
                };

                template <class Dim>
                struct get_storage<positional<Dim>> {
                    using type = positional<Dim>;
                };

                template <bool IsStateful>
                struct positionals : meta::list<positional<dim::i>, positional<dim::j>, positional<dim::k>> {};

                template <>
                struct positionals<false> : meta::list<> {};

            } // namespace lazy
            GT_META_DELEGATE_TO_LAZY(get_storage, class Arg, Arg);
            GT_META_DELEGATE_TO_LAZY(positionals, bool IsStateful, IsStateful);

            template <class Arg, class Src, class Dst>
            struct set_stride_f {
                Src const &m_src;
                Dst &m_dst;

                template <class Dim>
                void operator()() const {
                    at_key<Arg>(at_key<Dim>(m_dst)) = at_key<Dim>(m_src);
                }
            };
            template <class Arg, class Src, class Dst>
            set_stride_f<Arg, Src, Dst> set_stride(Src const &src, Dst &dst) {
                return {src, dst};
            }

            template <class Composite>
            struct x86_local_domain {
                GT_STATIC_ASSERT(is_sid<Composite>::value, GT_INTERNAL_ERROR);

                using ptr_holder_t = sid::ptr_holder_type<Composite>;
                using ptr_t = sid::ptr_type<Composite>;
                using strides_t = sid::strides_type<Composite>;
                using ptr_diff_t = sid::ptr_diff_type<Composite>;

                template <class Arg, class DataStore, std::enable_if_t<has_key<Composite, Arg>::value, int> = 0>
                void set_data_store(Arg, DataStore &data_store) {
                    GT_STATIC_ASSERT(is_sid<DataStore>::value, "");

                    at_key<Arg>(m_ptr_holder) = sid::get_origin(data_store);
                    using stride_dims_t = get_keys<sid::strides_type<DataStore>>;
                    auto const &src_strides = sid::get_strides(data_store);
                    for_each_type<stride_dims_t>(set_stride<Arg>(src_strides, m_strides));
                }

                template <class Arg, class DataStore, std::enable_if_t<!has_key<Composite, Arg>::value, int> = 0>
                void set_data_store(Arg, DataStore &) {}

                ptr_holder_t m_ptr_holder;
                strides_t m_strides;
            };

            template <class Mss, bool IsStateful>
            struct get_local_domain {
                using esf_args_t = extract_placeholders_from_mss<Mss>;
                GT_STATIC_ASSERT((meta::all_of<is_plh, esf_args_t>::value), GT_INTERNAL_ERROR);

                using positionals_t = positionals<IsStateful>;

                using args_t = meta::concat<esf_args_t, positionals_t>;

                using composite_keys_t = meta::rename<sid::composite::keys, args_t>;

                using storages_t = meta::transform<get_storage, args_t>;

                using composite_t = meta::rename<composite_keys_t::template values, storages_t>;

                using type = x86_local_domain<composite_t>;
            };
        } // namespace local_domain_impl_

        template <class Mss, bool IsStateful>
        using get_local_domain = typename local_domain_impl_::get_local_domain<Mss, IsStateful>::type;
    } // namespace x86
} // namespace gridtools
