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

#include "mss_local_domain.hpp"
#include "tile.hpp"

namespace gridtools {

    template < typename Backend,
        typename MssDescriptorArrayIn,
        typename DomainType,
        typename Grid,
        typename ConditionalsSet,
        typename ReductionType,
        bool IsStateful,
        uint_t RepeatFunctor = 1 >
    struct intermediate;

    template < typename T >
    struct is_intermediate : boost::mpl::false_ {};

    template < typename Backend,
        typename MssArray,
        typename DomainType,
        typename Grid,
        typename ConditionalsSet,
        typename ReductionType,
        bool IsStateful,
        uint_t RepeatFunctor >
    struct is_intermediate<
        intermediate< Backend, MssArray, DomainType, Grid, ConditionalsSet, ReductionType, IsStateful, RepeatFunctor > >
        : boost::mpl::true_ {};

    template < typename T >
    struct intermediate_backend;

    template < typename Backend,
        typename MssArray,
        typename DomainType,
        typename Grid,
        typename ConditionalsSet,
        typename ReductionType,
        bool IsStateful,
        uint_t RepeatFunctor >
    struct intermediate_backend< intermediate< Backend,
        MssArray,
        DomainType,
        Grid,
        ConditionalsSet,
        ReductionType,
        IsStateful,
        RepeatFunctor > > {
        typedef Backend type;
    };

    template < typename T >
    struct intermediate_aggregator_type;

    template < typename Backend,
        typename MssArray,
        typename DomainType,
        typename Grid,
        typename ConditionalsSet,
        typename ReductionType,
        bool IsStateful,
        uint_t RepeatFunctor >
    struct intermediate_aggregator_type< intermediate< Backend,
        MssArray,
        DomainType,
        Grid,
        ConditionalsSet,
        ReductionType,
        IsStateful,
        RepeatFunctor > > {
        typedef DomainType type;
    };

    template < typename T >
    struct intermediate_mss_array;

    template < typename Backend,
        typename MssArray,
        typename DomainType,
        typename Grid,
        typename ConditionalsSet,
        typename ReductionType,
        bool IsStateful,
        uint_t RepeatFunctor >
    struct intermediate_mss_array< intermediate< Backend,
        MssArray,
        DomainType,
        Grid,
        ConditionalsSet,
        ReductionType,
        IsStateful,
        RepeatFunctor > > {
        typedef MssArray type;
    };

    template < typename Intermediate >
    struct intermediate_mss_components_array {
        GRIDTOOLS_STATIC_ASSERT((is_intermediate< Intermediate >::value), GT_INTERNAL_ERROR);
        typedef typename Intermediate::mss_components_array_t type;
    };

    template < typename Intermediate >
    struct intermediate_extent_sizes {
        GRIDTOOLS_STATIC_ASSERT((is_intermediate< Intermediate >::value), GT_INTERNAL_ERROR);
        typedef typename Intermediate::extent_sizes_t type;
    };

    template < typename T >
    struct intermediate_layout_type;

    template < typename T >
    struct intermediate_is_stateful;

    template < typename Backend,
        typename MssArray,
        typename DomainType,
        typename Grid,
        typename ConditionalsSet,
        typename ReductionType,
        bool IsStateful,
        uint_t RepeatFunctor >
    struct intermediate_is_stateful< intermediate< Backend,
        MssArray,
        DomainType,
        Grid,
        ConditionalsSet,
        ReductionType,
        IsStateful,
        RepeatFunctor > > {
        typedef boost::mpl::bool_< IsStateful > type;
    };

    template < typename T >
    struct intermediate_mss_local_domains;

    template < typename Intermediate >
    struct intermediate_mss_local_domains {
        GRIDTOOLS_STATIC_ASSERT((is_intermediate< Intermediate >::value), GT_INTERNAL_ERROR);
        typedef typename Intermediate::mss_local_domains_t type;
    };

    namespace _impl {

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
                    // storage_info type that should be in the metadata_set
                    typedef gridtools::pointer< const typename StorageWrapper::storage_info_t > ms_storage_info_ptr_t;
                    // get the correct storage wrapper from the list of all storage wrappers
                    auto sw = boost::fusion::deref(boost::fusion::find< StorageWrapper >(m_storage_wrappers));
                    // feed the local domain with data ptrs
                    sw.assign(boost::fusion::deref(boost::fusion::find< typename StorageWrapper::arg_t >(
                                                       m_local_domain.m_local_data_ptrs))
                                  .second);
                    // feed the local domain with a storage info ptr
                    boost::fusion::deref(
                        boost::fusion::find< ld_storage_info_ptr_t >(m_local_domain.m_local_storage_info_ptrs)) =
                        Backend::template extract_storage_info_ptrs< ms_storage_info_ptr_t, AggregatorType >(
                            m_aggregator);
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

        /** @brief Functor used to delete all temporary storages */
        struct delete_tmp_data_store {
            template < typename T >
            void operator()(T &t) const {
                t.ptr.destroy();
            }
        };

        /** @brief Functor used to instantiate and allocate all temporary storages */
        template < typename MaxExtent, typename AggregatorType, typename Grid, typename Backend >
        struct instantiate_tmps {
            AggregatorType &m_agg;
            Grid const &m_grid;

            instantiate_tmps(AggregatorType &aggregator, Grid const &grid) : m_agg(aggregator), m_grid(grid) {}

            template < typename T, typename boost::enable_if_c< T::is_temporary, int >::type = 0 >
            void operator()(T const &) const {
                assert(!m_agg.template get_arg_storage_pair< typename T::arg_t >().ptr.get() &&
                       "temporary storage already initialized (maybe ready() was called multiple times). "
                       "reinitialization would produce a memory leak. ");
                // instantiate the right storage info (according to grid and used strategy)
                auto storage_info = Backend::template instantiate_storage_info< MaxExtent, T >(m_grid);
                // create a storage and fill the aggregator
                auto ptr = gridtools::pointer< typename T::storage_t >(new typename T::storage_t(storage_info));
                m_agg.template set_arg_storage_pair< typename T::arg_t >(ptr);
            }

            template < typename T, typename boost::enable_if_c< !T::is_temporary, int >::type = 1 >
            void operator()(T const &) const {}
        };

        /** @brief Functor used to synchronize all data_stores */
        struct sync_data_stores {
            // case for non temporary storages (perform sync)
            template < typename T >
            typename boost::enable_if_c< !is_arg_storage_pair_to_tmp< T >::value &&
                                             is_vector< typename T::storage_t >::value,
                void >::type
            operator()(T const &t) const {
                for (unsigned i = 0; i < t.ptr.get()->size(); ++i)
                    (*t.ptr)[i].sync();
            }

            // case for non temporary storages (perform sync)
            template < typename T >
            typename boost::enable_if_c< !is_arg_storage_pair_to_tmp< T >::value &&
                                             !is_vector< typename T::storage_t >::value,
                void >::type
            operator()(T const &t) const {
                t.ptr.get()->sync();
            }

            // temporary storages don't have to be synced.
            template < typename T >
            typename boost::enable_if_c< is_arg_storage_pair_to_tmp< T >::value, void >::type operator()(
                T const &t) const {}
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

        /**
           \brief defines a method which associates an
           tmp storage, whose extent depends on an index, to the
           element in the Temporaries vector at that index position.

           \tparam Temporaries is the vector of temporary placeholder types.
        */
        template < uint_t BI, uint_t BJ >
        struct get_storage_wrapper {
            template < typename MapElem >
            struct apply {
                typedef typename boost::mpl::second< MapElem >::type extent_t;
                typedef typename boost::mpl::first< MapElem >::type temporary;
                typedef storage_wrapper< temporary,
                    typename get_view_t::apply< typename temporary::storage_t >::type,
                    tile< BI, -extent_t::iminus::value, extent_t::iplus::value >,
                    tile< BJ, -extent_t::jminus::value, extent_t::jplus::value > > type;
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
                auto ds =
                    m_aggregator.template get_arg_storage_pair< typename boost::fusion::result_of::first< T >::type >();
                m_valid &= check_consistency(*(ds.ptr), v.second);
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

    } // namespace _impl

    namespace _debug {
        template < typename Grid >
        struct show_pair {
            Grid m_grid;

            explicit show_pair(Grid const &grid) : m_grid(grid) {}

            template < typename T >
            void operator()(T const &) const {
                typedef typename index_to_level< typename T::first >::type from;
                typedef typename index_to_level< typename T::second >::type to;
                std::cout << "{ (" << from() << " " << to() << ") "
                          << "[" << m_grid.template value_at< from >() << ", " << m_grid.template value_at< to >()
                          << "] } ";
            }
        };

    } // namespace _debug
} // namespace gridtools
