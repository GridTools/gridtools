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

#include <boost/fusion/include/at_key.hpp>
#include <boost/fusion/include/count.hpp>
#include <boost/fusion/include/flatten.hpp>
#include <boost/fusion/include/move.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/optional.hpp>

#include "../common/functional.hpp"
#include "../common/generic_metafunctions/copy_into_set.hpp"
#include "../common/tuple_util.hpp"
#include "../common/vector_traits.hpp"
#include "./tmp_storage.hpp"

#include "./extract_placeholders.hpp"
#include "./local_domain.hpp"

namespace gridtools {
    namespace _impl {

        /// this functor takes storage infos shared pointers (or types that contain storage_infos);
        /// stashes all infos that are passed through for the first time;
        /// if info is about to pass through twice, the functor substitutes it with the stashed one.
        template <class StorageInfoMap>
        struct dedup_storage_info_f {
            StorageInfoMap &m_storage_info_map;

            template <class Strorage, class StorageInfo>
            data_store<Strorage, StorageInfo> operator()(data_store<Strorage, StorageInfo> const &src) const {
                assert(src.valid());
                static_assert(boost::mpl::has_key<StorageInfoMap, StorageInfo>::value, "");
                auto &stored = boost::fusion::at_key<StorageInfo>(m_storage_info_map);
                if (!stored) {
                    stored = src.get_storage_info_ptr();
                    return src;
                }
                assert(*stored == *src.get_storage_info_ptr());
                return {src, stored};
            }

            template <class Storage, class StorageInfo>
            data_store<Storage, StorageInfo> operator()(data_store<Storage, StorageInfo> &&src) const {
                assert(src.valid());
                static_assert(boost::mpl::has_key<StorageInfoMap, StorageInfo>::value, "");
                auto &stored = boost::fusion::at_key<StorageInfo>(m_storage_info_map);
                if (!stored) {
                    stored = src.get_storage_info_ptr();
                    return src;
                }
                assert(*stored == *src.get_storage_info_ptr());
                return {std::move(src), *stored};
            }

            template <class DataStore, uint_t... N>
            data_store_field<DataStore, N...> operator()(data_store_field<DataStore, N...> src) const {
                for (auto &item : src.m_field)
                    item = this->operator()(item);
                return src;
            }

            template <class Arg, class DataStore>
            arg_storage_pair<Arg, DataStore> operator()(arg_storage_pair<Arg, DataStore> const &src) const {
                return this->operator()(src.m_value);
            }
            template <class Arg, class DataStore>
            arg_storage_pair<Arg, DataStore> operator()(arg_storage_pair<Arg, DataStore> &&src) const {
                return this->operator()(std::move(src.m_value));
            }
        };

        template <class Arg>
        struct get_storage_info {
            using type = typename Arg::data_store_t::storage_info_t;
        };

        template <class StorageInfo>
        struct get_storage_info_map_element {
            using type = boost::fusion::pair<StorageInfo, std::shared_ptr<StorageInfo>>;
        };

        template <typename Placeholders>
        using storage_info_map_t = typename boost::fusion::result_of::as_map<boost::mpl::transform_view<
            typename copy_into_set<boost::mpl::transform_view<Placeholders, get_storage_info<boost::mpl::_>>,
                boost::mpl::set0<>>::type,
            get_storage_info_map_element<boost::mpl::_>>>::type;

        template <typename Elem, access_mode AccessMode = access_mode::ReadWrite, typename Enable = void>
        struct get_view;

        template <typename Elem, access_mode AccessMode>
        struct get_view<Elem, AccessMode, typename boost::enable_if<is_data_store<Elem>>::type> {
            // we can use make_host_view here because the type is the
            // same for make_device_view and make_host_view.
            typedef decltype(make_host_view<AccessMode, Elem>(std::declval<Elem &>())) type;
        };

        template <typename Elem, access_mode AccessMode>
        struct get_view<Elem, AccessMode, typename boost::enable_if<is_data_store_field<Elem>>::type> {
            // we can use make_field_host_view here because the type is the
            // same for make_field_device_view and make_field_host_view.
            typedef decltype(make_field_host_view<AccessMode, Elem>(std::declval<Elem &>())) type;
        };

        template <class DataStorage>
        struct view_data {
            using view_t = typename get_view<DataStorage>::type;
            using storage_info_t = typename DataStorage::storage_info_t;

            boost::optional<view_t> m_view;
            storage_info_t m_storage_info;
        };

        /// This struct is used to hold bound storages. It holds a view.
        /// the method updated_view return creates a view only if the previously returned view was inconsistent.
        template <class Arg, class DataStorage>
        struct bound_arg_storage_pair {
            using view_t = typename get_view<DataStorage>::type;

            DataStorage m_data_storage;
            boost::optional<view_t> m_view;

            bound_arg_storage_pair(arg_storage_pair<Arg, DataStorage> const &src) : m_data_storage{src.m_value} {}
            bound_arg_storage_pair(arg_storage_pair<Arg, DataStorage> &&src) noexcept
                : m_data_storage{std::move(src.m_value)} {}

            template <class Backend>
            boost::optional<view_t> updated_view() {
                if (m_view && check_consistency(m_data_storage, *m_view))
                    return boost::none;
                if (m_data_storage.device_needs_update())
                    m_data_storage.sync();
                m_view.emplace(typename Backend::make_view_f{}(m_data_storage));
                return m_view;
            }
        };

        struct sync_f {
            template <class Arg, class DataStorage>
            void operator()(bound_arg_storage_pair<Arg, DataStorage> const &obj) const {
                obj.m_data_storage.sync();
            }
        };

        template <class Arg, class DataStorage>
        using view_info_t = boost::fusion::pair<Arg, view_data<DataStorage>>;

        template <class Backend>
        struct make_view_info_f {
            template <class Arg, class DataStorage>
            view_info_t<Arg, DataStorage> operator()(arg_storage_pair<Arg, DataStorage> const &src) const {
                const auto &storage = src.m_value;
                if (storage.device_needs_update())
                    storage.sync();

                return view_data<DataStorage>{
                    boost::make_optional(typename Backend::make_view_f{}(storage)), *storage.get_storage_info_ptr()};
            }
            template <class Arg, class DataStorage>
            view_info_t<Arg, DataStorage> operator()(bound_arg_storage_pair<Arg, DataStorage> &src) const {
                return view_data<DataStorage>{
                    src.template updated_view<Backend>(), *src.m_data_storage.get_storage_info_ptr()};
            }
        };

        template <class LocalDomain, class Arg>
        using local_domain_has_arg = typename boost::mpl::has_key<typename LocalDomain::data_ptr_fusion_map, Arg>::type;

        // set pointers from the given view info to the local domain
        struct set_view_to_local_domain_f {

            // if the arg belongs to the local domain we set pointers
            template <class Arg, class OptView, class LocalDomain>
            enable_if_t<local_domain_has_arg<LocalDomain, Arg>::value> operator()(
                boost::fusion::pair<Arg, OptView> const &info, LocalDomain &local_domain) const {
                if (!info.second.m_view)
                    return;
                auto const &view = *info.second.m_view;
                auto const &storage_info = info.second.m_storage_info;
                namespace f = boost::fusion;
                // here we set data pointers
                advanced::copy_raw_pointers(view, f::at_key<Arg>(local_domain.m_local_data_ptrs));
                // here we set meta data pointers
                auto const *storage_info_ptr = advanced::storage_info_raw_ptr(view);
                *f::find<decltype(storage_info_ptr)>(local_domain.m_local_storage_info_ptrs) = storage_info_ptr;
                // here we set strides
                using storage_info_t = remove_const_t<remove_reference_t<decltype(storage_info)>>;
                using index_t = meta::st_position<typename LocalDomain::storage_info_typelist, storage_info_t>;
                f::at_c<index_t::value>(local_domain.m_local_strides) = storage_info.strides();
            }
            // do nothing if arg is not in this local domain
            template <class Arg, class OptView, class LocalDomain>
            enable_if_t<!local_domain_has_arg<LocalDomain, Arg>::value> operator()(
                boost::fusion::pair<Arg, OptView> const &, LocalDomain &) const {}
        };

        template <class ViewInfos, class LocalDomains>
        void update_local_domains(ViewInfos const &view_infos, LocalDomains &local_domains) {
            tuple_util::for_each_in_cartesian_product(set_view_to_local_domain_f{}, view_infos, local_domains);
        }

        template <class MaxExtent, class Backend>
        struct get_tmp_arg_storage_pair_generator {
            template <class ArgStoragePair>
            struct generator {
                template <class Grid>
                ArgStoragePair operator()(Grid const &grid) const {
                    static constexpr auto backend = Backend{};
                    static constexpr auto arg = typename ArgStoragePair::arg_t{};
                    return make_tmp_data_store<MaxExtent>(backend, arg, grid);
                }
            };
            template <class T>
#if GT_BROKEN_TEMPLATE_ALIASES
            struct apply {
                using type = generator<T>;
            };
#else
            using apply = generator<T>;
#endif
        };

        template <class MaxExtent, class Backend, class Res, class Grid>
        Res make_tmp_arg_storage_pairs(Grid const &grid) {
            using generators = GT_META_CALL(
                meta::transform, (get_tmp_arg_storage_pair_generator<MaxExtent, Backend>::template apply, Res));
            return tuple_util::generate<generators, Res>(grid);
        }

        template <class Esf, class Extent>
        struct extent_for_tmp
            : std::conditional<boost::mpl::empty<typename esf_get_w_temps_per_functor<Esf>::type>::value,
                  extent<>,
                  Extent> {};

        template <class Extents>
        struct fold_extents : boost::mpl::fold<Extents, extent<>, enclosing_extent<boost::mpl::_1, boost::mpl::_2>> {};

        template <class MssComponents>
        struct get_max_extent_for_tmp_from_mss_components
            : fold_extents<typename boost::mpl::transform<typename MssComponents::linear_esf_t,
                  typename MssComponents::extent_sizes_t,
                  extent_for_tmp<boost::mpl::_1, boost::mpl::_2>>::type> {};

        template <class MssComponentsList>
        struct get_max_extent_for_tmp : fold_extents<boost::mpl::transform_view<MssComponentsList,
                                            get_max_extent_for_tmp_from_mss_components<boost::mpl::_>>> {};

        template <class MaxExtent, bool IsStateful>
        struct get_local_domain {
            template <class MssComponents, class Msses = std::tuple<typename MssComponents::mss_descriptor_t>>
            GT_META_DEFINE_ALIAS(
                apply, local_domain, (GT_META_CALL(extract_placeholders, Msses), MaxExtent, IsStateful));
        };

        template <class MssComponentsList,
            bool IsStateful,
            class MaxExtentForTmp = typename get_max_extent_for_tmp<MssComponentsList>::type,
            class GetLocalDomain = _impl::get_local_domain<MaxExtentForTmp, IsStateful>>
        GT_META_DEFINE_ALIAS(get_local_domains, meta::transform, (GetLocalDomain::template apply, MssComponentsList));
    } // namespace _impl
} // namespace gridtools
