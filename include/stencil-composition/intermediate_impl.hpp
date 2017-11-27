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

#include "common/generic_metafunctions/copy_into_set.hpp"
#include "common/permute_to.hpp"

#include "mss_local_domain.hpp"
#include "tile.hpp"

namespace gridtools {
    namespace _impl {

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
        };

        template < typename ArgStoragePairs, typename Res = get_storage_info_ptrs_t< ArgStoragePairs > >
        Res get_storage_info_ptrs(ArgStoragePairs const &src) {
            return permute_to< Res >(boost::fusion::as_vector(boost::fusion::transform(src, get_storage_info_ptr_f{})));
        }

        /** @brief Functor used to instantiate the local domains to be passed to each
            elementary stencil function */
        template < typename Backend, typename StorageWrapperFusionVec, typename AggregatorType, bool IsStateful >
        struct instantiate_local_domain {

            template < typename LocalDomain >
            struct assign_ptrs {
                StorageWrapperFusionVec &m_storage_wrappers;
                LocalDomain &m_local_domain;
                AggregatorType const &m_aggregator;
                assign_ptrs(
                    LocalDomain &ld, StorageWrapperFusionVec &storage_wrappers, AggregatorType const &aggregator)
                    : m_local_domain(ld), m_storage_wrappers(storage_wrappers), m_aggregator(aggregator) {}

                template < typename StorageWrapper >
                void operator()(StorageWrapper &t) const {
                    GRIDTOOLS_STATIC_ASSERT((is_storage_wrapper< StorageWrapper >::value), GT_INTERNAL_ERROR);
                    // storage_info type that should be in the local domain
                    typedef typename boost::add_pointer< typename boost::add_const<
                        typename StorageWrapper::storage_info_t >::type >::type ld_storage_info_ptr_t;
                    // get the correct storage wrapper from the list of all storage wrappers
                    auto sw = boost::fusion::deref(boost::fusion::find< StorageWrapper >(m_storage_wrappers));
                    // feed the local domain with data ptrs
                    sw.assign(boost::fusion::deref(boost::fusion::find< typename StorageWrapper::arg_t >(
                                                       m_local_domain.m_local_data_ptrs))
                                  .second);
                    boost::fusion::deref(
                        boost::fusion::find< ld_storage_info_ptr_t >(m_local_domain.m_local_storage_info_ptrs)) =
                        Backend::extract_storage_info_ptr(boost::fusion::at_key< ld_storage_info_ptr_t >(
                            get_storage_info_ptrs(m_aggregator.get_arg_storage_pairs())));
                }
            };

            GT_FUNCTION
            instantiate_local_domain(StorageWrapperFusionVec &storage_wrappers, AggregatorType const &aggregator)
                : m_storage_wrappers(storage_wrappers), m_aggregator(aggregator) {}

            /**Elem is a local_domain*/
            template < typename Elem >
            void operator()(Elem &elem) const {
                // set all the storage ptrs
                boost::mpl::for_each< typename Elem::storage_wrapper_list_t >(
                    assign_ptrs< Elem >(elem, m_storage_wrappers, m_aggregator));
                // clone the local domain to the device
                elem.clone_to_device();
            }

          private:
            StorageWrapperFusionVec &m_storage_wrappers;
            AggregatorType const &m_aggregator;
        };

        /** @brief Functor used to instantiate the mss local domains */
        template < typename Backend, typename StorageWrapperFusionVec, typename AggregatorType, bool IsStateful >
        struct instantiate_mss_local_domain {

            GT_FUNCTION
            instantiate_mss_local_domain(StorageWrapperFusionVec &storage_wrappers, AggregatorType const &aggregator)
                : m_storage_wrappers(storage_wrappers), m_aggregator(aggregator) {}

            /**Elem is a mss_local_domain*/
            template < typename Elem >
            GT_FUNCTION void operator()(Elem &mss_local_domain_list_) const {
                GRIDTOOLS_STATIC_ASSERT((is_mss_local_domain< Elem >::value), GT_INTERNAL_ERROR);
                boost::fusion::for_each(mss_local_domain_list_.local_domain_list,
                    _impl::instantiate_local_domain< Backend, StorageWrapperFusionVec, AggregatorType, IsStateful >(
                                            m_storage_wrappers, m_aggregator));
            }

          private:
            StorageWrapperFusionVec &m_storage_wrappers;
            AggregatorType const &m_aggregator;
        };

        /** @brief Functor used to instantiate and allocate all temporary storages */
        template < typename MaxExtent, typename AggregatorType, typename Grid, typename Backend >
        struct instantiate_tmps {
            AggregatorType &m_agg;
            Grid const &m_grid;

            instantiate_tmps(AggregatorType &aggregator, Grid const &grid) : m_agg(aggregator), m_grid(grid) {}

            template < typename T, typename boost::enable_if_c< T::is_temporary, int >::type = 0 >
            void operator()(T const &) const {
                // instantiate the right storage info (according to grid and used strategy)
                auto storage_info = Backend::template instantiate_storage_info< MaxExtent, T >(m_grid);
                // create a storage and fill the aggregator
                m_agg.template set_arg_storage_pair< typename T::arg_t >(typename T::data_store_t(storage_info));
            }

            template < typename T, typename boost::enable_if_c< !T::is_temporary, int >::type = 1 >
            void operator()(T const &) const {}
        };

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

        /** @brief Metafunction class used to get the view type */
        // TODO: ReadOnly support...
        struct get_view_t {
            template < typename Elem, access_mode AccessMode = access_mode::ReadWrite, typename Enable = void >
            struct apply;

            template < typename Elem, access_mode AccessMode >
            struct apply< Elem, AccessMode, typename boost::enable_if< is_data_store< Elem > >::type > {
                // we can use make_host_view here because the type is the
                // same for make_device_view and make_host_view.
                typedef decltype(make_host_view< AccessMode, Elem >(std::declval< Elem & >())) type;
            };

            template < typename Elem, access_mode AccessMode >
            struct apply< Elem, AccessMode, typename boost::enable_if< is_data_store_field< Elem > >::type > {
                // we can use make_field_host_view here because the type is the
                // same for make_field_device_view and make_field_host_view.
                typedef decltype(make_field_host_view< AccessMode, Elem >(std::declval< Elem & >())) type;
            };
        };

        /** @brief Functor used to check the consistency of all views */
        template < typename AggregatorType >
        struct check_view_consistency {
            AggregatorType const &m_aggregator;
            mutable bool m_valid;

            check_view_consistency(AggregatorType const &aggregator) : m_aggregator(aggregator), m_valid(true) {}

            template < typename T,
                typename boost::disable_if< is_tmp_arg< typename boost::fusion::result_of::first< T >::type >,
                    int >::type = 0 >
            void operator()(T const &v) const {
                const auto &ds =
                    m_aggregator.template get_arg_storage_pair< typename boost::fusion::result_of::first< T >::type >();
                m_valid &= check_consistency(ds.m_value, v.second);
            }

            template < typename T,
                typename boost::enable_if< is_tmp_arg< typename boost::fusion::result_of::first< T >::type >,
                    int >::type = 0 >
            void operator()(T const &v) const {}

            bool is_consistent() const { return m_valid; }
        };

        /** @brief Functor used to instantiate the storage_wrappers */
        template < typename Views >
        struct initialize_storage_wrappers {
            Views const &m_views;

            initialize_storage_wrappers(Views const &v) : m_views(v) {}

            template < typename T >
            void operator()(T &t) const {
                typedef typename arg_from_storage_wrapper< T >::type arg_t;
                t.initialize(boost::fusion::at_key< arg_t >(m_views));
            }
        };

        template < typename Backend >
        struct make_view_elem_f {
            template < typename A, typename DS >
            auto operator()(const arg_storage_pair< A, DS > &src) const
                GT_AUTO_RETURN(boost::fusion::make_pair< A >(Backend::make_view(src.m_value)));
        };

        template < typename Backend, typename Src, typename Dst >
        static void instantiate_views(const Src &src, Dst &dst) {
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of< Src, is_arg_storage_pair >::value), GT_INTERNAL_ERROR);
            boost::fusion::move(boost::fusion::transform(src, make_view_elem_f< Backend >{}), dst);
        }

        template < typename T1, typename T2 >
        struct matching {
            typedef typename boost::is_same< T1, T2 >::type type;
        };

        template < typename T1, typename T2 >
        struct contains {
            typedef typename boost::mpl::fold< T1,
                boost::mpl::false_,
                boost::mpl::or_< boost::mpl::_1, matching< boost::mpl::_2, T2 > > >::type type;
        };

        /**
         * @brief metafunction that computes the list of extents associated to each functor.
         * It assumes the temporary is written only by one esf.
         * TODO This assumption is probably wrong?, a temporary could be written my multiple esf concatenated. The
         * algorithm
         * we need to use here is find the maximum extent associated to a temporary instead.
         * @tparam TempsPerFunctor vector of vectors containing the list of temporaries written per esf
         * @tparam ExtendSizes extents associated to each esf (i.e. due to read access patterns of later esf's)
         */
        template < typename TMap, typename Temp, typename TempsPerFunctor, typename ExtendSizes >
        struct associate_extents_map {
            template < typename TTemp >
            struct is_temp_there {
                template < typename TempsInEsf >
                struct apply {
                    typedef typename contains< TempsInEsf, TTemp >::type type;
                };
            };

            typedef typename boost::mpl::find_if< TempsPerFunctor,
                typename is_temp_there< Temp >::template apply< boost::mpl::_ > >::type iter;

            typedef typename boost::mpl::if_<
                typename boost::is_same< iter, typename boost::mpl::end< TempsPerFunctor >::type >::type,
                TMap,
                typename boost::mpl::insert< TMap,
                    boost::mpl::pair< Temp, typename boost::mpl::at< ExtendSizes, typename iter::pos >::type > >::
                    type >::type type;
        };

        /**
         * @brief metafunction that computes the map of all the temporaries and their associated ij extents
         * @tparam AggregatorType domain type containing the placeholders for all storages (including temporaries)
         * @tparam MssComponents the mss components of the MSS
         * @output map of <temporary placeholder, extent> where the extent is the enclosing extent of all the extents
         *      defined for the different functors of a MSS.
         */
        template < typename AggregatorType, typename MssComponents >
        struct obtain_map_extents_temporaries_mss {
            GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< AggregatorType >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_mss_components< MssComponents >::value), GT_INTERNAL_ERROR);
            typedef typename MssComponents::extent_sizes_t ExtendSizes;

            // filter all the temporary args
            typedef typename boost::mpl::fold< typename AggregatorType::placeholders_t,
                boost::mpl::vector0<>,
                boost::mpl::if_< is_tmp_arg< boost::mpl::_2 >,
                                                   boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 >,
                                                   boost::mpl::_1 > >::type list_of_temporaries;

            // vector of written temporaries per functor (vector of vectors)
            typedef typename MssComponents::written_temps_per_functor_t written_temps_per_functor_t;

            typedef typename boost::mpl::fold< list_of_temporaries,
                boost::mpl::map0<>,
                associate_extents_map< boost::mpl::_1, boost::mpl::_2, written_temps_per_functor_t, ExtendSizes > >::
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
         * @tparam AggregatorType domain type containing the placeholders for all storages (including temporaries)
         * @tparam MssComponentsArray meta array of the mss components of all MSSs
         * @output map of <temporary placeholder, extent> where the extent is the enclosing extent of all the extents
         *      defined for the temporary in all MSSs.
         */
        template < typename AggregatorType, typename MssComponents >
        struct obtain_map_extents_temporaries_mss_array {
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of< MssComponents, is_mss_components >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< AggregatorType >::value), GT_INTERNAL_ERROR);

            typedef typename boost::mpl::fold<
                MssComponents,
                boost::mpl::map0<>,
                merge_extent_temporary_maps< boost::mpl::_1,
                    obtain_map_extents_temporaries_mss< AggregatorType, boost::mpl::_2 > > >::type type;
        };

        template < typename AggregatorType, typename MssArray1, typename MssArray2, typename Cond >
        struct obtain_map_extents_temporaries_mss_array< AggregatorType, condition< MssArray1, MssArray2, Cond > > {
            GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< AggregatorType >::value), GT_INTERNAL_ERROR);

            typedef typename obtain_map_extents_temporaries_mss_array< AggregatorType, MssArray1 >::type type1;
            typedef typename obtain_map_extents_temporaries_mss_array< AggregatorType, MssArray2 >::type type2;
            typedef
                typename boost::mpl::fold< type2, type1, boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > >::type
                    type;
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
                    typename get_view_t::apply< typename temporary::data_store_t >::type,
                    tile< BI, -extent_t::iminus::value, extent_t::iplus::value >,
                    tile< BJ, -extent_t::jminus::value, extent_t::jplus::value > > type;
            };
        };

        /**
         * @brief compute a list with all the storage_wrappers
         * @tparam AggregatorType domain
         * @tparam MssComponentsArray meta array of mss components
         */
        template < typename Backend, typename AggregatorType, typename MssComponents >
        struct obtain_storage_wrapper_list_t {

            GRIDTOOLS_STATIC_ASSERT(
                (is_condition< MssComponents >::value || is_sequence_of< MssComponents, is_mss_components >::value),
                GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< AggregatorType >::value), GT_INTERNAL_ERROR);

            using block_size_t = typename Backend::block_size_t;

            static const uint_t tileI = block_size_t::i_size_t::value;
            static const uint_t tileJ = block_size_t::j_size_t::value;

            typedef
                typename obtain_map_extents_temporaries_mss_array< AggregatorType, MssComponents >::type map_of_extents;

            typedef typename boost::mpl::fold<
                map_of_extents,
                boost::mpl::vector0<>,
                boost::mpl::push_back< boost::mpl::_1,
                    typename get_storage_wrapper< tileI, tileJ >::template apply< boost::mpl::_2 > > >::type type;
        };

    } // namespace _impl
} // namespace gridtools
