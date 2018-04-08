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
#include <boost/optional.hpp>

#include "common/generic_metafunctions/copy_into_set.hpp"
#include "common/functional.hpp"
#include "common/vector_traits.hpp"
#include "common/tuple_util.hpp"

#include "mss_local_domain.hpp"
#include "tile.hpp"

namespace gridtools {
    namespace _impl {

        /// this functor takes storage infos shared pointers (or types that contain storage_infos);
        /// stashes all infos that are passed through for the first time;
        /// if info is about to pass through twice, the functor substitutes it with the stashed one.
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

            template < class DataStore, uint_t... N >
            data_store_field< DataStore, N... > operator()(data_store_field< DataStore, N... > src) const {
                for (auto &item : src.m_field)
                    item = this->operator()(item);
                return src;
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

        /// This struct is used to hold bound storages. It holds a view.
        /// the method updated_view return creates a view only if the previously returned view was inconsistent.
        template < class Arg, class DataStorage >
        struct bound_arg_storage_pair {
            using view_t = typename get_view< DataStorage >::type;

            DataStorage m_data_storage;
            boost::optional< view_t > m_view;

            bound_arg_storage_pair(arg_storage_pair< Arg, DataStorage > const &src) : m_data_storage{src.m_value} {}
            bound_arg_storage_pair(arg_storage_pair< Arg, DataStorage > &&src) noexcept
                : m_data_storage{std::move(src.m_value)} {}

            template < class Backend >
            boost::optional< view_t > updated_view() {
                if (m_view && check_consistency(m_data_storage, *m_view))
                    return boost::none;
                m_data_storage.sync();
                m_view.emplace(typename Backend::make_view_f{}(m_data_storage));
                return m_view;
            }
        };

        struct sync_f {
            template < class Arg, class DataStorage >
            void operator()(bound_arg_storage_pair< Arg, DataStorage > const &obj) const {
                obj.m_data_storage.sync();
            }
        };

        template < class Arg, class DataStorage >
        using view_info_t = boost::fusion::pair< Arg, boost::optional< typename get_view< DataStorage >::type > >;

        template < class Backend >
        struct make_view_info_f {
            template < class Arg, class DataStorage >
            view_info_t< Arg, DataStorage > operator()(arg_storage_pair< Arg, DataStorage > const &src) const {
                src.m_value.sync();
                return boost::make_optional(typename Backend::make_view_f{}(src.m_value));
            }
            template < class Arg, class DataStorage >
            view_info_t< Arg, DataStorage > operator()(bound_arg_storage_pair< Arg, DataStorage > &src) const {
                return src.template updated_view< Backend >();
            }
        };

        template < class LocalDomain, class Arg >
        using local_domain_has_arg =
            typename boost::mpl::has_key< typename LocalDomain::data_ptr_fusion_map, Arg >::type;

        // set pointers from the given view info to the local domain
        template < class LocalDomain >
        struct set_view_to_local_domain_f {
            LocalDomain &m_local_domain;

            // if the arg belongs to the local domain we set pointers
            template < class Arg, class OptView >
            enable_if_t< local_domain_has_arg< LocalDomain, Arg >::value > operator()(
                boost::fusion::pair< Arg, OptView > const &info) const {
                if (!info.second)
                    return;
                auto const &view = *info.second;
                namespace f = boost::fusion;
                // here we set data pointers
                advanced::copy_raw_pointers(view, f::at_key< Arg >(m_local_domain.m_local_data_ptrs));
                // here we set meta data pointers
                auto const *storage_info = advanced::storage_info_ptr(view);
                *f::find< decltype(storage_info) >(m_local_domain.m_local_storage_info_ptrs) = storage_info;
            }
            // do nothing if arg is not in this local domain
            template < class Arg, class OptView >
            enable_if_t< !local_domain_has_arg< LocalDomain, Arg >::value > operator()(
                boost::fusion::pair< Arg, OptView > const &info) const {}
        };

        template < class ViewInfos >
        struct update_local_domain_f {
            ViewInfos const &m_view_infos;

            template < class LocalDomain >
            void operator()(LocalDomain &local_domain) const {
                tuple_util::for_each(set_view_to_local_domain_f< LocalDomain >{local_domain}, m_view_infos);
            }
        };

        struct get_local_domain_list_f {
            // Mind the double parens after GT_AUTO_RETURN. They are here for the reason.
            template < class T >
            auto operator()(T &&obj) const GT_AUTO_RETURN((std::forward< T >(obj).local_domain_list));
        };

        template < class ViewInfos, class MssLocalDomains >
        void update_local_domains(ViewInfos const &view_infos, MssLocalDomains &mss_local_domains) {
            // here we produce from mss_local_domains a flat tuple of references to local_domain;
            auto &&local_domains =
                tuple_util::flatten(tuple_util::transform(get_local_domain_list_f{}, mss_local_domains));
            // and for each possible local_domain/view_info pair call set_view_to_local_domain_f functor
            // TODO(anstaf): add for_each_cartesian_product to tuple_util and use it here.
            tuple_util::for_each(update_local_domain_f< ViewInfos >{view_infos}, std::move(local_domains));
        }

        template < class MaxExtent, class Backend, class StorageWrapperList >
        struct get_tmp_arg_storage_pair_generator {
            using tmp_storage_wrappers_t = typename boost::mpl::copy_if<
                StorageWrapperList,
                temporary_info_from_storage_wrapper< boost::mpl::_ >,
                boost::mpl::inserter<
                    boost::mpl::map0<>,
                    boost::mpl::insert< boost::mpl::_1,
                        boost::mpl::pair< arg_from_storage_wrapper< boost::mpl::_2 >, boost::mpl::_2 > > > >::type;
            template < class ArgStoragePair >
            struct generator {
                template < class Grid >
                ArgStoragePair operator()(Grid const &grid) const {
                    using arg_t = typename ArgStoragePair::arg_t;
                    using data_store_t = typename ArgStoragePair::data_store_t;
                    using storage_wrapper_t = typename boost::mpl::at< tmp_storage_wrappers_t, arg_t >::type;
                    return data_store_t{
                        Backend::template instantiate_storage_info< MaxExtent, storage_wrapper_t >(grid)};
                }
            };
            template < class T >
#if GT_BROKEN_TEMPLATE_ALIASES
            struct apply {
                using type = generator< T >;
            };
#else
            using apply = generator< T >;
#endif
        };

        template < class MaxExtent, class Backend, class StorageWrapperList, class Res, class Grid >
        Res make_tmp_arg_storage_pairs(Grid const &grid) {
            using generators = GT_META_CALL(meta::transform,
                (get_tmp_arg_storage_pair_generator< MaxExtent, Backend, StorageWrapperList >::template apply, Res));
            return tuple_util::generate< generators, Res >(grid);
        }

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
