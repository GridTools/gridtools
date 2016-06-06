#pragma once

#include <boost/mpl/filter_view.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/reverse.hpp>

#include "gridtools.hpp"
#include "stencil-composition/heap_allocated_temps.hpp"
#include "stencil-composition/backend_traits_fwd.hpp"
#include "stencil-composition/run_functor_arguments.hpp"

#ifdef __CUDACC__
#include "stencil-composition/backend_cuda/backend_cuda.hpp"
#else
#include "stencil-composition/backend_host/backend_host.hpp"
#endif

#include "common/pair.hpp"
#include "accessor.hpp"
#include "global_parameter.hpp"
#include "stencil-composition/domain_type.hpp"
#include "stencil-composition/mss_metafunctions.hpp"
#include "stencil-composition/mss_local_domain.hpp"
#include "stencil-composition/mss.hpp"
#include "stencil-composition/axis.hpp"
#include "common/meta_array.hpp"
#include "stencil-composition/tile.hpp"
#include "../storage/storage-facility.hpp"
#include "conditionals/condition.hpp"

/**
   @file
   @brief base class for all the backends. Current supported backend are \ref gridtools::enumtype::Host and \ref
   gridtools::enumtype::Cuda
   It is templated on the derived type (CRTP pattern) in order to use static polymorphism.
*/

namespace gridtools {

    template < typename T >
    struct is_meta_storage;

    namespace _impl {

        /**
           \brief defines a method which associates an
           tmp storage, whose extent depends on an index, to the
           element in the Temporaries vector at that index position.

           \tparam Temporaries is the vector of temporary placeholder types.
        */
        template < typename TemporariesExtendMap,
            typename ValueType,
            uint_t BI,
            uint_t BJ,
            typename StrategyTraits,
            enumtype::platform BackendID >
        struct get_storage_type {
            template < typename MapElem >
            struct apply {
                typedef typename boost::mpl::second< MapElem >::type extent_t;
                typedef typename boost::mpl::first< MapElem >::type temporary;

                typedef pair_type< typename StrategyTraits::template get_tmp_storage< typename temporary::storage_type,
                                       tile< BI, -extent_t::iminus::value, extent_t::iplus::value >,
                                       tile< BJ, -extent_t::jminus::value, extent_t::jplus::value > >::type,
                    typename temporary::index_type > type;
            };
        };
    } // namespace _impl

    /** metafunction to check whether the storage_type inside the PlcArgType is temporary */
    template < typename PlcArgType >
    struct is_temporary_arg : is_temporary_storage< typename PlcArgType::storage_type > {};

    /**
        this struct contains the 'run' method for all backends, with a
        policy determining the specific type. Each backend contains a
        traits class for the specific case.

        backend<type, strategy>
        there are traits: one for type and one for strategy.
        - type refers to the architecture specific, like the
          differences between cuda and the host.

        The backend has a member function "run" that is called by the
        "intermediate".
        The "run" method calls strategy_from_id<strategy>::loop

        - the strategy_from_id is in the specific backend_? folder, such as
        - in backend_?/backend_traits.h

        - strategy_from_id contains the tile size information and the
        - "struct loop" which has the "run_loop" member function.

        Before calling the loop::run_loop method, the backend queries
        "execute_traits" that are contained in the
        "backend_traits_t". the latter is obtained by

        backend_strategy_from_id<type>

        The execute_traits::backend_t (bad name) is responsible for
        the "inner loop nests". The
        loop<execute_traits::backend_t>::run_loop will use that to do
        whatever he has to do, for instance, the host_backend will
        iterate over the functors of the MSS using the for_each
        available there.

        - Similarly, the definition (specialization) is contained in the
        - specific subfoled (right now in backend_?/backend_traits_?.h ).

        - This contains:
        - - (INTERFACE) pointer<>::type that returns the first argument to instantiate the storage class
        - - (INTERFACE) storage_traits::storage_t to get the storage type to be used with the backend
        - - (INTERFACE) execute_traits ?????? this was needed when backend_traits was forcely shared between host and
       cuda backends. Now they are separated and this may be simplified.
        - - (INTERNAL) for_each that is used to invoke the different things for different stencils in the MSS
        - - (INTERNAL) once_per_block
    */
    template < enumtype::platform BackendId, enumtype::grid_type GridId, enumtype::strategy StrategyId >
    struct backend_base {
        typedef backend_base< BackendId, GridId, StrategyId > this_type;

        typedef backend_ids< BackendId, GridId, StrategyId > backend_ids_t;

        typedef backend_traits_from_id< BackendId > backend_traits_t;
        typedef grid_traits_from_id< GridId > grid_traits_t;
        typedef typename backend_traits_t::template select_strategy< backend_ids_t >::type strategy_traits_t;

        static const enumtype::strategy s_strategy_id = StrategyId;
        static const enumtype::platform s_backend_id = BackendId;
        static const enumtype::grid_type s_grid_type_id = GridId;

        /** types of the functions used to compute the thread grid information
            for allocating the temporary storages and such
        */
        typedef uint_t (*query_i_threads_f)(uint_t);
        typedef uint_t (*query_j_threads_f)(uint_t);

        template < typename ValueType, typename MetaDataType >
        struct storage_type {
            typedef typename storage_traits<BackendId>::storage_traits_aux::template select_storage<ValueType,
                typename storage_traits<BackendId>::storage_traits_aux::template select_meta_storage< typename MetaDataType::index_type,
                                                                    typename MetaDataType::layout,
                                                                    false,
                                                                    typename MetaDataType::halo_t,
                                                                    typename MetaDataType::alignment_t>::type
            >::type type;
        };

#ifdef CXX11_ENABLED

        /**
           @brief syntactic sugar for the metadata type definition

           \tparam Index an index used to differentiate the types also when there's only runtime
           differences (e.g. only the storage dimensions differ)
           \tparam Layout the map of the layout in memory
           \tparam IsTemporary boolean flag set to true when the storage is a temporary one
           \tmaram ... Tiles variadic argument containing the information abount the tiles
           (for the Block strategy)

           syntax example:
           using metadata_t=storage_info<0,layout_map<0,1,2> >

           NOTE: the information specified here will be used at a later stage
           to define the storage meta information (the meta_storage_base type)
        */
        template < ushort_t Index,
            typename Layout,
            typename Halo = typename repeat_template_c< 0, Layout::length, halo >::type,
            typename Alignment = typename storage_traits<BackendId>::storage_traits_aux::default_alignment::type >
        using storage_info = typename storage_traits<BackendId>::storage_traits_aux::
            template select_meta_storage< static_uint< Index >, Layout, false, Halo, Alignment >::type;

#else
        template < ushort_t Index,
            typename Layout,
            typename Halo = halo< 0, 0, 0 >,
            typename Alignment = typename storage_traits<BackendId>::storage_traits_aux::default_alignment::type >
        struct storage_info
            : public storage_traits<BackendId>::storage_traits_aux::
                  template select_meta_storage< static_uint< Index >, Layout, false, Halo, Alignment >::type {
            typedef typename storage_traits<BackendId>::storage_traits_aux::
                template select_meta_storage< static_uint< Index >, Layout, false, Halo, Alignment >::type super;

            storage_info(uint_t const &d1, uint_t const &d2, uint_t const &d3) : super(d1, d2, d3) {}

            GT_FUNCTION
            storage_info(storage_info const &t) : super(t) {}
        };

#endif

        /**
         * @brief metafunction determining the type of a temporary storage (based on the layout)
         * If the backend fuses multiple ESFs of a computation, it will require applying redundant computation
         * at some halo points of each block. In this case a "no_storage_type_yet" type is selected, which will
         * be replace into an actual storage allocating enough space for the redundant halo points. In this case,
         * the allocated space will depend on block sizes and extents of the ESF (that is why we need to delay the
         * instantiation of the actual storage type). If on the contrary multiple ESFs are not fused, a "standard"
         * storage type will be enough.
         */
        template < typename ValueType, typename MetaDataType >
        struct temporary_storage_type {
            GRIDTOOLS_STATIC_ASSERT(is_meta_storage< MetaDataType >::value, "wrong type for the meta storage");
            /** temporary storage must have the same iterator type than the regular storage
             */
          private:
            typedef typename storage_traits<BackendId>::storage_traits_aux::template select_storage< ValueType,
                typename storage_traits<BackendId>::storage_traits_aux::template select_meta_storage< typename MetaDataType::index_type,
                                                                            typename MetaDataType::layout,
                                                                            true,
                                                                            typename MetaDataType::halo_t,
                                                                            typename MetaDataType::alignment_t >::type
            >::type temp_storage_t;

          public:
            typedef no_storage_type_yet< temp_storage_t > type;
        };

        /**
         * @brief metafunction that computes the map of all the temporaries and their associated ij extents
         * @tparam Domain domain type containing the placeholders for all storages (including temporaries)
         * @tparam MssComponents the mss components of the MSS
         * @output map of <temporary placeholder, extent> where the extent is the enclosing extent of all the extents
         *      defined for the different functors of a MSS.
         */
        template < typename Domain, typename MssComponents >
        struct obtain_map_extents_temporaries_mss {
            GRIDTOOLS_STATIC_ASSERT((is_domain_type< Domain >::value), "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT((is_mss_components< MssComponents >::value), "Internal Error: wrong type");
            typedef typename MssComponents::extent_sizes_t ExtendSizes;

            typedef typename _impl::extract_temporaries< typename Domain::placeholders >::type list_of_temporaries;

            // vector of written temporaries per functor (vector of vectors)
            typedef typename MssComponents::written_temps_per_functor_t written_temps_per_functor_t;

            typedef typename boost::mpl::fold< list_of_temporaries,
                boost::mpl::map0<>,
                _impl::associate_extents_map< boost::mpl::_1,
                                                   boost::mpl::_2,
                                                   written_temps_per_functor_t,
                                                   ExtendSizes > >::type type;
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
         * @tparam Domain domain type containing the placeholders for all storages (including temporaries)
         * @tparam MssComponentsArray meta array of the mss components of all MSSs
         * @output map of <temporary placeholder, extent> where the extent is the enclosing extent of all the extents
         *      defined for the temporary in all MSSs.
         */
        template < typename Domain, typename MssComponentsArray >
        struct obtain_map_extents_temporaries_mss_array {
            GRIDTOOLS_STATIC_ASSERT(
                (is_meta_array_of< MssComponentsArray, is_mss_components >::value), "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT((is_domain_type< Domain >::value), "Internal Error: wrong type");

            typedef
                typename boost::mpl::fold< typename MssComponentsArray::elements,
                    boost::mpl::map0<>,
                    merge_extent_temporary_maps< boost::mpl::_1,
                                               obtain_map_extents_temporaries_mss< Domain, boost::mpl::_2 > > >::type
                    type;
        };

        template < typename Domain, typename MssArray1, typename MssArray2, typename Cond >
        struct obtain_map_extents_temporaries_mss_array< Domain, condition< MssArray1, MssArray2, Cond > > {
            GRIDTOOLS_STATIC_ASSERT((is_domain_type< Domain >::value), "Internal Error: wrong type");

            typedef typename obtain_map_extents_temporaries_mss_array< Domain, MssArray1 >::type type1;
            typedef typename obtain_map_extents_temporaries_mss_array< Domain, MssArray2 >::type type2;
            typedef
                typename boost::mpl::fold< type2, type1, boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > >::type
                    type;
        };

        /**
         * @brief compute a list with all the temporary storage types used by an array of mss
         * @tparam Domain domain
         * @tparam MssComponentsArray meta array of mss components
         * @tparam ValueType type of field values stored in the temporary storage
         * @tparam LayoutType memory layout
         */
        template < typename Domain, typename MssComponentsArray, typename ValueType >
        struct obtain_temporary_storage_types {

            GRIDTOOLS_STATIC_ASSERT((is_condition< MssComponentsArray >::value ||
                                        is_meta_array_of< MssComponentsArray, is_mss_components >::value),
                "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT((is_domain_type< Domain >::value), "Internal Error: wrong type");

            typedef typename backend_traits_t::template get_block_size< StrategyId >::type block_size_t;

            static const uint_t tileI = block_size_t::i_size_t::value;
            static const uint_t tileJ = block_size_t::j_size_t::value;

            typedef boost::mpl::filter_view< typename Domain::placeholders, is_temporary_arg< boost::mpl::_ > >
                temporaries;
            typedef
                typename obtain_map_extents_temporaries_mss_array< Domain, MssComponentsArray >::type map_of_extents;

            // GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<temporaries>::value ==
            // boost::mpl::size<map_of_extents>::value),
            //         "One of the temporaries was not found in at least one functor of all the MSS.\n Check that all
            //         temporaries declared as in the domain are actually used in at least a functor"
            // )

            typedef typename boost::mpl::fold<
                map_of_extents,
                boost::mpl::vector<>,
                typename boost::mpl::push_back< typename boost::mpl::_1,
                    typename _impl::
                        get_storage_type< map_of_extents, ValueType, tileI, tileJ, strategy_traits_t, s_backend_id >::
                            template apply< boost::mpl::_2 > > >::type type;
        };

        /**
         * \brief calls the \ref gridtools::run_functor for each functor in the FunctorList.
         * the loop over the functors list is unrolled at compile-time using the for_each construct.
         * @tparam MssArray  meta array of mss
         * \tparam Domain Domain class (not really useful maybe)
         * \tparam Grid Coordinate class with domain sizes and splitter grid
         * \tparam MssLocalDomainArray sequence of mss local domain (containing each the sequence of local domain list)
         */
        template < typename MssComponentsArray,
            typename Grid,
            typename MssLocalDomainArray,
            typename ReductionData > // List of local domain to be pbassed to functor at<i>
        static void
        run(/*Domain const& domain, */ Grid const &grid,
            MssLocalDomainArray &mss_local_domain_list,
            ReductionData &reduction_data) {
            // TODO: I would swap the arguments coords and local_domain_list here, for consistency
            GRIDTOOLS_STATIC_ASSERT(
                (is_sequence_of< MssLocalDomainArray, is_mss_local_domain >::value), "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT(
                (is_meta_array_of< MssComponentsArray, is_mss_components >::value), "Internal Error: wrong type");

            strategy_traits_t::template fused_mss_loop< MssComponentsArray, backend_ids_t, ReductionData >::run(
                mss_local_domain_list, grid, reduction_data);
        }

        template < typename ArgList, typename MetaList, typename Grid >
        static void prepare_temporaries(ArgList &arg_list_, MetaList &meta_list_, Grid const &grid) {
            _impl::template prepare_temporaries_functor< ArgList,
                MetaList,
                Grid,
                backend_ids_t>::prepare_temporaries((arg_list_), meta_list_, (grid));
        }

        /** Initial interface

            Threads are oganized in a 2D grid. These two functions
            n_i_pes() and n_j_pes() retrieve the
            information about how to compute those sizes.

            The information needed by those functions are the sizes of the
            domains (especially if the GPU is used)

            n_i_pes()(size): number of threads on the first dimension of the thread grid
        */
        static query_i_threads_f n_i_pes() { return &backend_traits_t::n_i_pes; }

        /** Initial interface

            Threads are oganized in a 2D grid. These two functions
            n_i_pes() and n_j_pes() retrieve the
            information about how to compute those sizes.

            The information needed by those functions are the sizes of the
            domains (especially if the GPU is used)

            n_j_pes()(size): number of threads on the second dimension of the thread grid
        */
        static query_j_threads_f n_j_pes() { return &backend_traits_t::n_j_pes; }

    }; // struct backend_base {

    template < template < ushort_t, typename, typename, typename > class StorageInfo,
        ushort_t Index,
        typename Layout,
        typename Halo,
        typename Alignment >
    struct is_meta_storage< StorageInfo< Index, Layout, Halo, Alignment > > : boost::mpl::true_ {};

} // namespace gridtools
