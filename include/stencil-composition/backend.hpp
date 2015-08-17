#pragma once

#include <boost/mpl/filter_view.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/reverse.hpp>
#include <boost/mpl/zip_view.hpp>

#include <gridtools.hpp>

#include "backend_traits_fwd.hpp"
#include "run_functor_arguments.hpp"

#ifdef __CUDACC__
#include "stencil-composition/backend_cuda/backend_cuda.hpp"
#else
#include "stencil-composition/backend_host/backend_host.hpp"
#endif

#include "../common/pair.hpp"
#include "heap_allocated_temps.hpp"
#include "accessor.hpp"
#include "domain_type.hpp"
#include "mss_metafunctions.hpp"
#include "mss_local_domain.hpp"
#include "mss.hpp"
#include "axis.hpp"
#include "../common/meta_array.hpp"

/**
   @file
   @brief base class for all the backends. Current supported backend are \ref gridtools::enumtype::Host and \ref gridtools::enumtype::Cuda
   It is templated on the derived type (CRTP pattern) in order to use static polymorphism.
*/

namespace gridtools {

    namespace _impl {

        /**
           \brief defines a method which associates an
           host_tmp_storage, whose range depends on an index, to the
           element in the Temporaries vector at that index position.

           \tparam Temporaries is the vector of temporary placeholder types.
        */
        template <typename TemporariesRangeMap,
                  typename ValueType,
                  typename LayoutType,
                  uint_t BI, uint_t BJ,
                  typename StrategyTraits,
                  enumtype::backend BackendID>
        struct get_storage_type {
            template <typename MapElem>
            struct apply {
                typedef typename boost::mpl::second<MapElem>::type range_type;
                typedef typename boost::mpl::first<MapElem>::type temporary;

                typedef pair<
                    typename StrategyTraits::template get_tmp_storage<
                    typename temporary::storage_type::type,
                    BI, BJ,
                    -range_type::iminus::value,
                    -range_type::jminus::value,
                    range_type::iplus::value,
                    range_type::jplus::value>::type, // previously : host_storage_t,
                    typename temporary::index_type
                    > type;
            };
        };
    } // namespace _impl


    /** metafunction to check whether the storage_type inside the PlcArgType is temporary */
    template <typename PlcArgType>
    struct is_temporary_arg : is_temporary_storage<typename PlcArgType::storage_type>{};



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
        - - (INTERFACE) execute_traits ?????? this was needed when backend_traits was forcely shared between host and cuda backends. Now they are separated and this may be simplified.
        - - (INTERNAL) for_each that is used to invoke the different things for different stencils in the MSS
        - - (INTERNAL) once_per_block
    */
    template< enumtype::backend BackendId, enumtype::strategy StrategyType >
    struct backend
    {
        typedef backend_traits_from_id<BackendId> backend_traits_t;
        typedef typename backend_traits_t::template select_strategy<StrategyType>::type strategy_traits_t;

        typedef backend<BackendId, StrategyType> this_type;
        static const enumtype::strategy s_strategy_id=StrategyType;
        static const enumtype::backend s_backend_id =BackendId;

        /** types of the functions used to compute the thread grid information
            for allocating the temporary storages and such
        */
        typedef uint_t (*query_i_threads_f)(uint_t);
        typedef uint_t (*query_j_threads_f)(uint_t);

        template <typename ValueType, typename Layout>
        struct storage_type {
            typedef typename backend_traits_t::template storage_traits<ValueType, Layout>::storage_t type;
        };

        /**
         * @brief metafunction determining the type of a temporary storage (based on the layout)
         * If the backend fuses multiple ESFs of a computation, it will require applying redundant computation
         * at some halo points of each block. In this case a "no_storage_type_yet" type is selected, which will
         * be replace into an actual storage allocating enough space for the redundant halo points. In this case,
         * the allocated space will depend on block sizes and ranges of the ESF (that is why we need to delay the
         * instantiation of the actual storage type). If on the contrary multiple ESFs are not fused, a "standard"
         * storage type will be enough.
         */
        template <typename ValueType, typename Layout>
        struct temporary_storage_type
        {
            /** temporary storage must have the same iterator type than the regular storage
             */
        private:
            typedef typename backend_traits_t::template storage_traits<ValueType, Layout, true>::storage_t temp_storage_t;
        public:
            typedef typename boost::mpl::if_<
                typename backend_traits_t::template requires_temporary_redundant_halos<s_strategy_id>::type,
                no_storage_type_yet< temp_storage_t >,
                temp_storage_t
            >::type type;
        };


        /**
         * @brief metafunction that computes the map of all the temporaries and their associated ij ranges
         * @tparam Domain domain type containing the placeholders for all storages (including temporaries)
         * @tparam MssComponents the mss components of the MSS
         * @output map of <temporary placeholder, range> where the range is the enclosing range of all the ranges
         *      defined for the different functors of a MSS.
         */
        template <
            typename Domain,
            typename MssComponents>
        struct obtain_map_ranges_temporaries_mss
        {
            GRIDTOOLS_STATIC_ASSERT((is_domain_type<Domain>::value), "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT((is_mss_components<MssComponents>::value), "Internal Error: wrong type");
            typedef typename MssComponents::range_sizes_t RangeSizes;

            typedef typename _impl::extract_temporaries<typename Domain::placeholders>::type list_of_temporaries;

            //vector of written temporaries per functor (vector of vectors)
            typedef typename MssComponents::written_temps_per_functor_t written_temps_per_functor_t;

            typedef typename boost::mpl::fold<
                list_of_temporaries,
                boost::mpl::map0<>,
                _impl::associate_ranges_map<boost::mpl::_1, boost::mpl::_2, written_temps_per_functor_t, RangeSizes>
            >::type type;
        };

        /**
         * @brief metafunction that merges two maps of <temporary, ij range>
         * The merge is performed by computing the union of all the ranges found associated
         * to the same temporary, i.e. the enclosing range.
         * @tparam range_map1 first map to merge
         * @tparam range_map2 second map to merge
          */
        template<typename range_map1, typename range_map2>
        struct merge_range_temporary_maps
        {
            typedef typename boost::mpl::fold<
                range_map1,
                range_map2,
                boost::mpl::if_<
                    boost::mpl::has_key<range_map2, boost::mpl::first<boost::mpl::_2> >,
                    boost::mpl::insert<
                        boost::mpl::_1,
                        boost::mpl::pair<
                            boost::mpl::first<boost::mpl::_2>,
                            enclosing_range<
                                boost::mpl::second<boost::mpl::_2>,
                                boost::mpl::at<range_map2, boost::mpl::first<boost::mpl::_2> >
                            >
                        >
                    >,
                    boost::mpl::insert<
                        boost::mpl::_1,
                        boost::mpl::_2
                    >
                >
            >::type type;
        };

        /**
         * @brief metafunction that computes the map of all the temporaries and their associated ij ranges
         * for all the Mss components in an array (corresponding to a Computation)
         * @tparam Domain domain type containing the placeholders for all storages (including temporaries)
         * @tparam MssComponentsArray meta array of the mss components of all MSSs
         * @output map of <temporary placeholder, range> where the range is the enclosing range of all the ranges
         *      defined for the temporary in all MSSs.
         */
        template <typename Domain, typename MssComponentsArray>
        struct obtain_map_ranges_temporaries_mss_array {
            GRIDTOOLS_STATIC_ASSERT((is_meta_array_of<MssComponentsArray, is_mss_components>::value), "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT((is_domain_type<Domain>::value), "Internal Error: wrong type");

            typedef typename boost::mpl::fold<
                typename MssComponentsArray::elements,
                boost::mpl::map0<>,
                merge_range_temporary_maps<
                    boost::mpl::_1,
                    obtain_map_ranges_temporaries_mss<Domain, boost::mpl::_2>
                >
            >::type type;
        };

        /**
         * @brief compute a list with all the temporary storage types used by an array of mss
         * @tparam Domain domain
         * @tparam MssComponentsArray meta array of mss components
         * @tparam ValueType type of field values stored in the temporary storage
         * @tparam LayoutType memory layout
         */
        template <typename Domain
                  , typename MssComponentsArray
                  , typename ValueType
                  , typename LayoutType >
        struct obtain_temporary_storage_types {

            GRIDTOOLS_STATIC_ASSERT((is_meta_array_of<MssComponentsArray, is_mss_components>::value), "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT((is_domain_type<Domain>::value), "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT((is_layout_map<LayoutType>::value), "Internal Error: wrong type");

            typedef typename backend_traits_t::template get_block_size<StrategyType>::type block_size_t;

            static const uint_t tileI = block_size_t::i_size_t::value;
            static const uint_t tileJ = block_size_t::j_size_t::value;

            typedef boost::mpl::filter_view<typename Domain::placeholders,
                                            is_temporary_arg<boost::mpl::_> > temporaries;
            typedef typename obtain_map_ranges_temporaries_mss_array<Domain, MssComponentsArray>::type map_of_ranges;


            // GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<temporaries>::value == boost::mpl::size<map_of_ranges>::value),
            //         "One of the temporaries was not found in at least one functor of all the MSS.\n Check that all temporaries declared as in the domain are actually used in at least a functor"
            // )

            typedef typename boost::mpl::fold<
                map_of_ranges,
                boost::mpl::vector<>,
                typename boost::mpl::push_back<
                    typename boost::mpl::_1,
                    typename _impl::get_storage_type<
                        map_of_ranges,
                        ValueType,
                        LayoutType,
                        tileI,
                        tileJ,
                        strategy_traits_t,
                        s_backend_id
                    >::template apply<boost::mpl::_2>
                >
            >::type type;
        };




        /**
         * \brief calls the \ref gridtools::run_functor for each functor in the FunctorList.
         * the loop over the functors list is unrolled at compile-time using the for_each construct.
         * @tparam MssArray  meta array of mss
         * \tparam Domain Domain class (not really useful maybe)
         * \tparam Coords Coordinate class with domain sizes and splitter coordinates
         * \tparam MssLocalDomainArray sequence of mss local domain (containing each the sequence of local domain list)
         */
        template <
            typename MssComponentsArray,
            typename Coords,
            typename MssLocalDomainArray
        > // List of local domain to be pbassed to functor at<i>
        static void run(/*Domain const& domain, */Coords const& coords, MssLocalDomainArray &mss_local_domain_list) {
            // TODO: I would swap the arguments coords and local_domain_list here, for consistency
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of<MssLocalDomainArray, is_mss_local_domain>::value), "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT((is_coordinates<Coords>::value), "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT((is_meta_array_of<MssComponentsArray, is_mss_components>::value), "Internal Error: wrong type");

            strategy_traits_t::template fused_mss_loop<MssComponentsArray, BackendId>::run(mss_local_domain_list, coords);
        }


        template <typename ArgList, typename Coords>
        static void prepare_temporaries(ArgList & arg_list, Coords const& coords)
        {
            _impl::template prepare_temporaries_functor<ArgList, Coords, this_type>::
                prepare_temporaries((arg_list), (coords));
        }

        /** Initial interface

            Threads are oganized in a 2D grid. These two functions
            n_i_pes() and n_j_pes() retrieve the
            information about how to compute those sizes.

            The information needed by those functions are the sizes of the
            domains (especially if the GPU is used)

            n_i_pes()(size): number of threads on the first dimension of the thread grid
        */
        static query_i_threads_f n_i_pes() {
            return &backend_traits_t::n_i_pes;
        }

        /** Initial interface

            Threads are oganized in a 2D grid. These two functions
            n_i_pes() and n_j_pes() retrieve the
            information about how to compute those sizes.

            The information needed by those functions are the sizes of the
            domains (especially if the GPU is used)

            n_j_pes()(size): number of threads on the second dimension of the thread grid
        */
        static query_j_threads_f n_j_pes() {
            return &backend_traits_t::n_j_pes;
        }


    }; // struct backend {

} // namespace gridtools
