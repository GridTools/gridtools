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

#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/move.hpp>
#include <boost/fusion/include/flatten.hpp>
#include <boost/fusion/include/count.hpp>

#include "common/generic_metafunctions/copy_into_set.hpp"
#include "common/fusion.hpp"
#include "common/permute_to.hpp"
#include "common/functional.hpp"

#include "mss_local_domain.hpp"
#include "tile.hpp"

namespace gridtools {
    namespace _impl {

        template < class StorageInfoMap >
        struct dedup_storage_info_f {
            StorageInfoMap &m_storage_info_map;

            template < class Strorage, class StorageInfo >
            data_store< Strorage, StorageInfo > operator()(data_store< Strorage, StorageInfo > const &src) const {
                assert(src.valid());
                static_assert(boost::mpl::has_key< StorageInfoMap, StorageInfo >::value, "");
                auto &stored = boost::fusion::at_key< StorageInfo >(m_storage_info_map);
                if (!stored) {
                    stored = src.get_storage_info_ptr();
                    return src;
                }
                assert(*stored == *src.get_storage_info_ptr());
                return {src, stored};
            }

            template < class Storage, class StorageInfo >
            data_store< Storage, StorageInfo > operator()(data_store< Storage, StorageInfo > &&src) const {
                assert(src.valid());
                static_assert(boost::mpl::has_key< StorageInfoMap, StorageInfo >::value, "");
                auto &stored = boost::fusion::at_key< StorageInfo >(m_storage_info_map);
                if (!stored) {
                    stored = src.get_storage_info_ptr();
                    return src;
                }
                assert(*stored == *src.get_storage_info_ptr());
                return {std::move(src), *stored};
            }

            template < class Arg, class DataStore >
            arg_storage_pair< Arg, DataStore > operator()(arg_storage_pair< Arg, DataStore > const &src) const {
                return this->operator()(src.m_value);
            }
            template < class Arg, class DataStore >
            arg_storage_pair< Arg, DataStore > operator()(arg_storage_pair< Arg, DataStore > &&src) const {
                return this->operator()(std::move(src.m_value));
            }
        };

        template < class Arg >
        struct get_storage_info {
            using type = typename Arg::data_store_t::storage_info_t;
        };

        template < class StorageInfo >
        struct get_storage_info_map_element {
            using type = boost::fusion::pair< StorageInfo, std::shared_ptr< StorageInfo > >;
        };

        template < typename Placeholders >
        using storage_info_map_t = typename boost::fusion::result_of::as_map< boost::mpl::transform_view<
            typename copy_into_set< boost::mpl::transform_view< Placeholders, get_storage_info< boost::mpl::_ > >,
                boost::mpl::set0<> >::type,
            get_storage_info_map_element< boost::mpl::_ > > >::type;

        template < typename ArgStoragePair >
        struct get_meta_ptr {
            using type = typename ArgStoragePair::arg_t::data_store_t::storage_info_t const *;
        };

        template < typename ArgStoragePairs >
        using get_storage_info_ptrs_t = typename boost::fusion::result_of::as_set<
            typename copy_into_set< boost::mpl::transform_view< ArgStoragePairs, get_meta_ptr< boost::mpl::_ > >,
                boost::mpl::set0<> >::type >::type;

        struct get_data_store_f {
            template < typename S >
            S const &operator()(S const &src) const {
                return src;
            }
            template < typename S, uint_t... N >
            S const &operator()(data_store_field< S, N... > const &src) const {
                return src.template get< 0, 0 >();
            }
        };

        struct get_storage_info_ptr_f {
            template < typename Src >
            auto operator()(Src const &src) const
                GT_AUTO_RETURN(get_data_store_f{}(src.m_value).get_storage_info_ptr().get());
#ifdef BOOST_RESULT_OF_USE_TR1
            template < typename >
            struct result;
            template < typename A, typename DS >
            struct result< get_storage_info_ptr_f(arg_storage_pair< A, DS >) > {
                using data_store_t =
                    typename std::decay< typename std::result_of< get_data_store_f(DS const &) >::type >::type;
                using type = typename data_store_t::storage_info_t const *;
            };
            template < typename Src >
            struct result< get_storage_info_ptr_f(Src &) >
                : result< get_storage_info_ptr_f(typename std::decay< Src >::type) > {};
#endif
        };

        template < typename ArgStoragePairs, typename Res = get_storage_info_ptrs_t< ArgStoragePairs > >
        Res get_storage_info_ptrs(ArgStoragePairs const &src) {
            return permute_to< Res >(boost::fusion::as_vector(boost::fusion::transform(src, get_storage_info_ptr_f{})));
        }

        template < class Backend, class Arg, class LocalDomain, class View >
        void set_view_to_local_domain(LocalDomain &local_domain, View const &view) {
            namespace f = boost::fusion;
            advanced::copy_raw_pointers(view, f::at_key< Arg >(local_domain.m_local_data_ptrs));
            *f::find< typename View::storage_info_t const * >(local_domain.m_local_storage_info_ptrs) =
                typename Backend::extract_storage_info_ptr_f{}(&view.storage_info());
        }

        template < typename Elem, access_mode AccessMode = access_mode::ReadWrite, typename Enable = void >
        struct get_view;

        template < typename Elem, access_mode AccessMode >
        struct get_view< Elem, AccessMode, typename boost::enable_if< is_data_store< Elem > >::type > {
            // we can use make_host_view here because the type is the
            // same for make_device_view and make_host_view.
            typedef decltype(make_host_view< AccessMode, Elem >(std::declval< Elem & >())) type;
        };

        template < typename Elem, access_mode AccessMode >
        struct get_view< Elem, AccessMode, typename boost::enable_if< is_data_store_field< Elem > >::type > {
            // we can use make_field_host_view here because the type is the
            // same for make_field_device_view and make_field_host_view.
            typedef decltype(make_field_host_view< AccessMode, Elem >(std::declval< Elem & >())) type;
        };

        template < class LocalDomain, class Arg >
        using local_domain_has_arg =
            typename boost::mpl::has_key< typename LocalDomain::data_ptr_fusion_map, Arg >::type;

        template < class Arg, class DataStorage >
        struct bound_arg_storage_pair {
            DataStorage m_data_storage;
            typename get_view< DataStorage >::type m_view = {};

            bound_arg_storage_pair(arg_storage_pair< Arg, DataStorage > const &src) : m_data_storage{src.m_value} {}
            bound_arg_storage_pair(arg_storage_pair< Arg, DataStorage > &&src) noexcept
                : m_data_storage{std::move(src.m_value)} {}

            template < class Backend >
            bool update_view() {
                bool need_update = !check_consistency(m_data_storage, m_view);
                if (need_update) {
                    m_data_storage.sync();
                    m_view = typename Backend::make_view_f{}(m_data_storage);
                }
                return need_update;
            }
        };

        struct sync_f {
            template < class Arg, class DataStorage >
            void operator()(bound_arg_storage_pair< Arg, DataStorage > &obj) const {
                obj.m_data_storage.sync();
            }
        };

        template < class Arg, class View >
        struct view_info {
            View view;
            bool enable_update;
        };

        template < class Backend >
        struct make_view_info_f {
            template < class Arg, class DataStorage >
            view_info< Arg, typename get_view< DataStorage >::type > operator()(
                arg_storage_pair< Arg, DataStorage > const &src) const {
                src.m_value.sync();
                return {typename Backend::make_view_f{}(src.m_value), true};
            }
            template < class Arg, class DataStorage >
            view_info< Arg, typename get_view< DataStorage >::type const & > operator()(
                bound_arg_storage_pair< Arg, DataStorage > &src) const {
                bool enable_update = src.template update_view< Backend >();
                return {src.m_view, enable_update};
            }
            template < class Arg, class DataStorage >
            view_info< Arg, typename get_view< DataStorage >::type const & > operator()(
                bound_arg_storage_pair< Arg, DataStorage > &&src) const;
        };

        template < class Backend, class LocalDomain >
        struct set_view_to_local_domain_f {
            LocalDomain &m_local_domain;
            template < class Arg, class View >
            typename std::enable_if< local_domain_has_arg< LocalDomain, Arg >::value, bool >::type operator()(
                view_info< Arg, View > const &info) const {
                if (info.enable_update)
                    set_view_to_local_domain< Backend, Arg >(m_local_domain, info.view);
                return info.enable_update;
            }
            template < class Arg, class View >
            typename std::enable_if< !local_domain_has_arg< LocalDomain, Arg >::value, bool >::type operator()(
                view_info< Arg, View > const &info) const {
                return false;
            }
        };

        template < class Backend, class ViewInfos >
        struct update_local_domain_f {
            ViewInfos const &m_view_infos;

            template < class LocalDomain >
            void operator()(LocalDomain &local_domain) const {
                namespace f = boost::fusion;
                if (f::count(
                        f::transform(m_view_infos, set_view_to_local_domain_f< Backend, LocalDomain >{local_domain}),
                        true))
                    local_domain.clone_to_device();
            }
        };

        struct add_ref_f {
            template < class T >
            typename std::add_lvalue_reference< T >::type operator()(T &&obj) const {
                return obj;
            }
        };

        struct get_local_domain_ref_list_f {
            template < class T >
            auto operator()(T &&obj) const
                GT_AUTO_RETURN(make_transform_view(std::forward< T >(obj).local_domain_list, add_ref_f{}));
        };

        template < class T >
        auto flatten_mss_local_domains(T &&src) GT_AUTO_RETURN(
            boost::fusion::flatten(make_transform_view(std::forward< T >(src), get_local_domain_ref_list_f{})));

        template < class Backend, class ViewInfos, class MssLocalDomains >
        void update_local_domains(ViewInfos const &view_infos, MssLocalDomains &mss_local_domains) {
            auto vec = boost::fusion::as_vector(view_infos);
            boost::fusion::for_each(
                flatten_mss_local_domains(mss_local_domains), update_local_domain_f< Backend, decltype(vec) >{vec});
        }

        template < class MaxExtent, class Backend, class StorageWrapperList, class Grid >
        struct tmp_arg_storage_pair_generator_f {
            using tmp_storage_wrappers_t = typename boost::mpl::copy_if<
                StorageWrapperList,
                temporary_info_from_storage_wrapper< boost::mpl::_ >,
                boost::mpl::inserter<
                    boost::mpl::map0<>,
                    boost::mpl::insert< boost::mpl::_1,
                        boost::mpl::pair< arg_from_storage_wrapper< boost::mpl::_2 >, boost::mpl::_2 > > > >::type;
            Grid const &m_grid;
            template < class ArgStoragePair >
            ArgStoragePair operator()() const {
                using arg_t = typename ArgStoragePair::arg_t;
                using data_store_t = typename ArgStoragePair::data_store_t;
                using storage_wrapper_t = typename boost::mpl::at< tmp_storage_wrappers_t, arg_t >::type;
                return data_store_t{Backend::template instantiate_storage_info< MaxExtent, storage_wrapper_t >(m_grid)};
            }
        };

        template < class MaxExtent, class Backend, class StorageWrapperList, class Res, class Grid >
        Res make_tmp_arg_storage_pairs(Grid const &grid) {
            return generate_sequence< Res >(
                tmp_arg_storage_pair_generator_f< MaxExtent, Backend, StorageWrapperList, Grid >{grid});
        }

        /** @brief Functor used to synchronize all data_stores */
        struct sync_data_stores {
            // case for non temporary storages (perform sync)
            template < typename T >
            typename boost::enable_if_c< !is_tmp_arg< T >::value && is_vector< typename T::data_store_t >::value,
                void >::type
            operator()(T const &t) const {
                for (auto &&item : t.m_value)
                    item.sync();
            }

            // case for non temporary storages (perform sync)
            template < typename T >
            typename boost::enable_if_c< !is_tmp_arg< T >::value && !is_vector< typename T::data_store_t >::value,
                void >::type
            operator()(T const &t) const {
                t.m_value.sync();
            }

            // temporary storages don't have to be synced.
            template < typename T >
            typename boost::enable_if_c< is_tmp_arg< T >::value, void >::type operator()(T const &t) const {}
        };

        /**
         * @brief metafunction that computes the list of extents associated to each functor.
         * It assumes the temporary is written only by one esf.
         * TODO This assumption is probably wrong?, a temporary could be written my multiple esf concatenated. The
         * algorithm
         * we need to use here is find the maximum extent associated to a temporary instead.
         * @tparam TempsPerFunctor vector of vectors containing the list of temporaries written per esf
         * @tparam ExtentSizes extents associated to each esf (i.e. due to read access patterns of later esf's)
         */
        template < typename TMap, typename Temp, typename TempsPerFunctor, typename ExtentSizes >
        struct associate_extents_map {
            template < typename TTemp >
            struct is_temp_there {
                template < typename TempsInEsf >
                struct apply {
                    typedef typename boost::mpl::contains< TempsInEsf, TTemp >::type type;
                };
            };

            typedef typename boost::mpl::find_if< TempsPerFunctor,
                typename is_temp_there< Temp >::template apply< boost::mpl::_ > >::type iter;

            typedef typename boost::mpl::if_<
                typename boost::is_same< iter, typename boost::mpl::end< TempsPerFunctor >::type >::type,
                TMap,
                typename boost::mpl::insert< TMap,
                    boost::mpl::pair< Temp, typename boost::mpl::at< ExtentSizes, typename iter::pos >::type > >::
                    type >::type type;
        };

        /**
         * @brief metafunction that computes the map of all the temporaries and their associated ij extents
         * @tparam Placeholders the placeholders for all storages (including temporaries)
         * @tparam MssComponents the mss components of the MSS
         * @output map of <temporary placeholder, extent> where the extent is the enclosing extent of all the extents
         *      defined for the different functors of a MSS.
         */
        template < typename Placeholders, typename MssComponents >
        struct obtain_map_extents_temporaries_mss {
            GRIDTOOLS_STATIC_ASSERT((is_mss_components< MssComponents >::value), GT_INTERNAL_ERROR);
            typedef typename MssComponents::extent_sizes_t ExtentSizes;

            // filter all the temporary args
            typedef typename boost::mpl::fold< Placeholders,
                boost::mpl::vector0<>,
                boost::mpl::if_< is_tmp_arg< boost::mpl::_2 >,
                                                   boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 >,
                                                   boost::mpl::_1 > >::type list_of_temporaries;

            // vector of written temporaries per functor (vector of vectors)
            typedef typename MssComponents::written_temps_per_functor_t written_temps_per_functor_t;

            typedef typename boost::mpl::fold< list_of_temporaries,
                boost::mpl::map0<>,
                associate_extents_map< boost::mpl::_1, boost::mpl::_2, written_temps_per_functor_t, ExtentSizes > >::
                type type;
        };

        /**
         * @brief metafunction that merges two maps of <temporary, ij extent>
         * The merge is performed by computing the union of all the extents found associated
         * to the same temporary, i.e. the enclosing extent.
         * @tparam extent_map1 first map to merge
         * @tparam extent_map2 second map to merge
          */
        template < typename extent_map1, typename extent_map2 >
        struct merge_extent_temporary_maps {
            typedef typename boost::mpl::fold<
                extent_map1,
                extent_map2,
                boost::mpl::if_< boost::mpl::has_key< extent_map2, boost::mpl::first< boost::mpl::_2 > >,
                    boost::mpl::insert< boost::mpl::_1,
                                     boost::mpl::pair< boost::mpl::first< boost::mpl::_2 >,
                                            enclosing_extent< boost::mpl::second< boost::mpl::_2 >,
                                                           boost::mpl::at< extent_map2,
                                                                  boost::mpl::first< boost::mpl::_2 > > > > >,
                    boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > > >::type type;
        };

        /**
         * @brief metafunction that computes the map of all the temporaries and their associated ij extents
         * for all the Mss components in an array (corresponding to a Computation)
         * @tparam Placeholders the placeholders for all storages (including temporaries)
         * @tparam MssComponentsArray meta array of the mss components of all MSSs
         * @output map of <temporary placeholder, extent> where the extent is the enclosing extent of all the extents
         *      defined for the temporary in all MSSs.
         */
        template < typename Placeholders, typename MssComponents >
        struct obtain_map_extents_temporaries_mss_array {
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of< MssComponents, is_mss_components >::value), GT_INTERNAL_ERROR);

            typedef typename boost::mpl::fold<
                MssComponents,
                boost::mpl::map0<>,
                merge_extent_temporary_maps< boost::mpl::_1,
                    obtain_map_extents_temporaries_mss< Placeholders, boost::mpl::_2 > > >::type type;
        };

        /**
           \brief defines a method which associates an
           tmp storage, whose extent depends on an index, to the
           element in the Temporaries vector at that index position.
        */
        template < uint_t BI, uint_t BJ >
        struct get_storage_wrapper {
            template < typename MapElem >
            struct apply {
                typedef typename boost::mpl::second< MapElem >::type extent_t;
                typedef typename boost::mpl::first< MapElem >::type temporary;
                typedef storage_wrapper< temporary,
                    typename get_view< typename temporary::data_store_t >::type,
                    tile< BI, -extent_t::iminus::value, extent_t::iplus::value >,
                    tile< BJ, -extent_t::jminus::value, extent_t::jplus::value > > type;
            };
        };

        /**
         * @brief compute a list with all the storage_wrappers
         * @tparam AggregatorType domain
         * @tparam MssComponentsArray meta array of mss components
         */
        template < typename Backend, typename Placeholders, typename MssComponents >
        struct obtain_storage_wrapper_list_t {

            GRIDTOOLS_STATIC_ASSERT((is_sequence_of< MssComponents, is_mss_components >::value), GT_INTERNAL_ERROR);

            using block_size_t = typename Backend::block_size_t;

            static const uint_t tileI = block_size_t::i_size_t::value;
            static const uint_t tileJ = block_size_t::j_size_t::value;

            typedef
                typename obtain_map_extents_temporaries_mss_array< Placeholders, MssComponents >::type map_of_extents;

            typedef typename boost::mpl::fold<
                map_of_extents,
                boost::mpl::vector0<>,
                boost::mpl::push_back< boost::mpl::_1,
                    typename get_storage_wrapper< tileI, tileJ >::template apply< boost::mpl::_2 > > >::type type;
        };

    } // namespace _impl
} // namespace gridtools
