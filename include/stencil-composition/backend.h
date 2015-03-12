#pragma once

#include <boost/mpl/filter_view.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/reverse.hpp>
#include <boost/mpl/zip_view.hpp>

#include "backend_traits.h"
#include "../common/pair.h"
#include "heap_allocated_temps.h"
#include "arg_type.h"
#include "domain_type.h"
#include "execution_types.h"
#include "mss_metafunctions.h"
#include "mss.h"
#include "axis.h"
#include "../common/meta_array.h"

/**
   @file
   @brief base class for all the backends. Current supported backend are \ref gridtools::enumtype::Host and \ref gridtools::enumtype::Cuda
   It is templated on the derived type (CRTP pattern) in order to use static polymorphism.
*/

namespace gridtools {

    namespace _impl {
/**
   \brief "base" struct for all the backend
   This class implements static polimorphism by means of the CRTP pattern. It contains all what is common for all the backends.
*/
    template < typename Derived >
    struct run_functor {

        typedef Derived derived_t;
        typedef run_functor_traits<Derived> derived_traits_t;
        typedef typename derived_traits_t::arguments_t arguments_t;
        typedef typename derived_traits_t::local_domain_list_t local_domain_list_t;
        typedef typename derived_traits_t::coords_t coords_t;

        local_domain_list_t & m_local_domain_list;
        coords_t const & m_coords;
        const uint_t m_starti, m_startj, m_BI, m_BJ, blk_idx_i, blk_idx_j;

        // Block strategy
        explicit run_functor(local_domain_list_t& dom_list, coords_t const& coords,
                uint_t i, uint_t j, uint_t bi, uint_t bj, uint_t blk_idx_i, uint_t blk_idx_j) :
            m_local_domain_list(dom_list), m_coords(coords), m_starti(i), m_startj(j), m_BI(bi),
            m_BJ(bj), blk_idx_i(blk_idx_i), blk_idx_j(blk_idx_j){}

        // Naive strategy
        explicit run_functor(local_domain_list_t& dom_list, coords_t const& coords) :
            m_local_domain_list(dom_list), m_coords(coords), m_starti(coords.i_low_bound()),
            m_startj(coords.j_low_bound()), m_BI(coords.i_high_bound()-coords.i_low_bound()),
            m_BJ(coords.j_high_bound()-coords.j_low_bound()), blk_idx_i(0), blk_idx_j(0) {}

        /**
         * \brief given the index of a functor in the functors list ,it calls a kernel on the GPU executing the operations defined on that functor.
         */
        template <typename Index>
        void operator()(Index const& ) const {

            typename derived_traits_t::template traits<Index>::local_domain_t& local_domain = boost::fusion::at<Index>(m_local_domain_list);
            typedef execute_kernel_functor<  derived_t > exec_functor_t;

            //check that the number of placeholders passed to the elementary stencil function
            //(constructed during the computation) is the same as the number of arguments referenced
            //in the functor definition (in the high level interface). This means that we cannot
            // (although in theory we could) pass placeholders to the computation which are not
            //also referenced in the functor.
            GRIDTOOLS_STATIC_ASSERT(boost::mpl::size<typename derived_traits_t::template traits<Index>::local_domain_t::esf_args>::value==
                    boost::mpl::size<typename derived_traits_t::template traits<Index>::functor_t::arg_list>::value,
		            "GRIDTOOLS ERROR:\n\
		            check that the number of placeholders passed to the elementary stencil function\n \
		            (constructed during the computation) is the same as the number of arguments referenced\n\
		            in the functor definition (in the high level interface). This means that we cannot\n\
		            (although in theory we could) pass placeholders to the computation which are not\n\
		            also referenced in the functor.");

		    exec_functor_t::template execute_kernel< typename derived_traits_t::template traits<Index> >(local_domain, static_cast<const derived_t*>(this));
        }
    };

    /**
        \brief defines a method which associates an host_tmp_storage, whose range depends on an index, to the element in the Temporaries vector at that index position.
         \tparam Temporaries is the vector of temporary placeholder types.
     */
    template <typename TemporaryRangeMap, typename ValueType, typename LayoutType, uint_t BI, uint_t BJ, typename StrategyTraits, enumtype::backend BackendID>
    struct get_storage_type {
        template <typename MapElem>
        struct apply {
            typedef typename boost::mpl::second<MapElem>::type range_type_t;
            typedef typename boost::mpl::first<MapElem>::type temporary_t;
            typedef pair<
                    typename StrategyTraits::template tmp<BackendID, ValueType, LayoutType, BI, BJ, -range_type_t::iminus::value, -range_type_t::jminus::value, range_type_t::iplus::value, range_type_t::jplus::value>::host_storage_t,
                    typename temporary_t::index_type
            > type;
        };
    };


/** metafunction to check whether the storage_type inside the PlcArgType is temporary */
    template <typename PlcArgType>
    struct is_temporary_arg : is_temporary_storage<typename PlcArgType::storage_type>{};

    }//namespace _impl

/** this struct contains the 'run' method for all backends, with a policy determining the specific type. Each backend contains a traits class for the specific case. */
    template< enumtype::backend BackendId, enumtype::strategy StrategyType >
    struct backend: public heap_allocated_temps<backend<BackendId, StrategyType > >
    {
        typedef backend_traits_from_id <BackendId> backend_traits_t;
        typedef strategy_from_id <StrategyType> strategy_traits_t;
        static const enumtype::strategy s_strategy_id=StrategyType;
        static const enumtype::backend s_backend_id =BackendId;

        template <typename ValueType, typename Layout>
        struct storage_type {
            typedef typename backend_traits_t::template storage_traits<ValueType, Layout>::storage_t type;
        };


        template <typename ValueType, typename Layout>
        struct temporary_storage_type
        {
            // temporary storage must have the same iterator type than the regular storage
            private:
                typedef typename backend_traits_t::template storage_traits<ValueType, Layout, true>::storage_t temp_storage_t;
            public:
                typedef typename boost::mpl::if_<typename boost::mpl::bool_<s_strategy_id==enumtype::Naive>::type,
                                       temp_storage_t,
                                       no_storage_type_yet< temp_storage_t > >::type type;
        };

        template<typename T> struct printk{BOOST_MPL_ASSERT_MSG((false), YYYYYYYYYY, (T));};

        /**
         * @brief it generates a map of temporaries and their associate range
         * @tparam Domain domain that contains the placeholders
         * @tparam MssType mss descriptor type
         */
        template <typename Domain
                  , typename MssType>
        struct obtain_map_ranges_temporaries_mss
        {
            typedef typename MssType::range_sizes RangeSizes;
            //full list of temporaries in list of place holders of domain
            typedef typename boost::mpl::fold<typename Domain::placeholders,
                boost::mpl::vector<>,
                boost::mpl::if_<
                    is_plchldr_to_temp<boost::mpl::_2>,
                    boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2 >,
                    boost::mpl::_1>
            >::type list_of_temporaries;

            //vector of written temporaries per functor (vector of vectors)
            typedef typename MssType::written_temps_per_functor written_temps_per_functor;

            typedef typename boost::mpl::fold<
                list_of_temporaries,
                boost::mpl::map0<>,
                _impl::associate_ranges_map<boost::mpl::_1, boost::mpl::_2, written_temps_per_functor, RangeSizes>
            >::type type;
        };


        /**
         * @brief metafunction that merges to maps of <temporary, ij range>
         * @tparam RangeMap1 first map to merge
         * @tparam RangeMap2 second map to merge
          */
        template<typename RangeMap1, typename RangeMap2>
        struct merge_range_temporary_maps
        {
            template<typename Map, typename Pair>
            struct compute_union_and_insert
            {
                typedef typename boost::mpl::first<Pair>::type key_t;
                BOOST_STATIC_ASSERT((boost::mpl::has_key<Map, key_t>::value));
                typedef typename boost::mpl::at<Map, key_t>::type OrigPair;
                typedef boost::mpl::insert<
                    typename boost::mpl::erase_key<Map, key_t>::type,
                    boost::mpl::pair<
                        key_t,
                        typename union_ranges<
                            typename boost::mpl::second<Pair>::type,
                            OrigPair
                        >::type
                    >
                > type;
            };

            template<typename Map, typename Pair>
            struct lazy_insert_in_map
            {
                BOOST_STATIC_ASSERT((!boost::mpl::has_key<Map, typename boost::mpl::first<Pair>::type>::value));
                typedef boost::mpl::insert<Map, Pair> type;
            };

            typedef typename boost::mpl::fold<
                RangeMap1,
                RangeMap2,
                boost::mpl::eval_if<
                    boost::mpl::has_key<RangeMap2, boost::mpl::first<boost::mpl::_2> >,
                    compute_union_and_insert<boost::mpl::_1, boost::mpl::_2>,
                    lazy_insert_in_map<boost::mpl::_1, boost::mpl::_2>
                >
            >::type type;
        };

        /**
         * @brief metafunction that computes the map of all the temporaries and their associated ij ranges
         */
        template <typename Domain, typename MssArray>
        struct obtain_map_ranges_temporaries_mss_array {
            BOOST_STATIC_ASSERT((is_meta_array_of<MssArray, is_mss_descriptor>::value));
            BOOST_STATIC_ASSERT((is_domain_type<Domain>::value));

            typedef typename boost::mpl::fold<
                typename MssArray::elements_t,
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
         * @tparam MssArray meta array of mss
         * @tparam ValueType type of field values stored in the temporary storage
         * @tparam LayoutType memory layout
         */
        template <typename Domain
                  , typename MssArray
                  , typename ValueType
                  , typename LayoutType >
        struct obtain_temporary_storage_types {

            BOOST_STATIC_ASSERT((is_meta_array_of<MssArray, is_mss_descriptor>::value));
            BOOST_STATIC_ASSERT((is_domain_type<Domain>::value));
            BOOST_STATIC_ASSERT((is_layout_map<LayoutType>::value));

            static const uint_t tileI = (strategy_traits_t::BI);

            static const uint_t tileJ = (strategy_traits_t::BJ);

            typedef boost::mpl::filter_view<typename Domain::placeholders, _impl::is_temporary_arg<boost::mpl::_> > temporaries;
            typedef typename obtain_map_ranges_temporaries_mss_array<Domain, MssArray>::type map_of_ranges;

            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<temporaries>::value == boost::mpl::size<map_of_ranges>::value),
                    "One of the temporaries was not found in at least one functor of all the MSS.\n Check that all temporaries declared as in the domain are actually used in at least a functor"
            );

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
            typename MssArray,
            typename Coords,
            typename MssLocalDomainArray
        > // List of local domain to be pbassed to functor at<i>
        static void run(/*Domain const& domain, */Coords const& coords, MssLocalDomainArray &mss_local_domain_list) {
            // TODO: I would swap the arguments coords and local_domain_list here, for consistency
            BOOST_STATIC_ASSERT((is_sequence_of<MssLocalDomainArray, is_mss_local_domain>::value));
            BOOST_STATIC_ASSERT((is_coordinates<Coords>::value));
            BOOST_STATIC_ASSERT((is_meta_array_of<MssArray, is_mss_descriptor>::value));

            strategy_from_id< s_strategy_id >::template fused_mss_loop<MssArray, BackendId>::run(mss_local_domain_list, coords);
        }


        template <typename ArgList, typename Coords>
        static void prepare_temporaries(ArgList & arg_list, Coords const& coords)
        {
            _impl::template prepare_temporaries_functor<ArgList, Coords, s_strategy_id>::prepare_temporaries(/*std::forward<ArgList&>*/(arg_list), /*std::forward<Coords const&>*/(coords));
        }
    };


} // namespace gridtools
