#pragma once

#include <boost/mpl/filter_view.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/reverse.hpp>

#include "backend_traits.h"
#include "../common/pair.h"
#include "heap_allocated_temps.h"
#include "arg_type.h"
#include "domain_type.h"
#include "execution_types.h"

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

            typename derived_traits_t::domain_list_t & m_domain_list;
            typename derived_traits_t::coords_t const & m_coords;
            const uint_t m_starti, m_startj, m_BI, m_BJ, blk_idx_i, blk_idx_j;

            // Block strategy
            explicit run_functor(typename derived_traits_t::domain_list_t& dom_list, typename derived_traits_t::coords_t const& coords, uint_t i, uint_t j, uint_t bi, uint_t bj, uint_t blk_idx_i, uint_t blk_idx_j)
                : m_domain_list(dom_list)
                , m_coords(coords)
                , m_starti(i)
                , m_startj(j)
                , m_BI(bi)
                , m_BJ(bj)
                , blk_idx_i(blk_idx_i)
                , blk_idx_j(blk_idx_j)
            {}

            // Naive strategy
            explicit run_functor(typename derived_traits_t::domain_list_t& dom_list, typename derived_traits_t::coords_t const& coords)
                :
                m_domain_list(dom_list)
                , m_coords(coords)
                , m_starti(coords.i_low_bound())
                , m_startj(coords.j_low_bound())
                , m_BI(coords.i_high_bound()-coords.i_low_bound())
                , m_BJ(coords.j_high_bound()-coords.j_low_bound())
                , blk_idx_i(0)
                , blk_idx_j(0)
            {}

            /**
             * \brief given the index of a functor in the functors list ,it calls a kernel on the GPU executing the operations defined on that functor.
             */
            template <typename Index>
            void operator()(Index const& ) const {

                typename derived_traits_t::template traits<Index>::local_domain_t& local_domain = boost::fusion::at<Index>(m_domain_list);
                typedef execute_kernel_functor<  derived_t > exec_functor_t;

		//check that the number of placeholders passed to the elementary stencil function
		//(constructed during the computation) is the same as the number of arguments referenced
		//in the functor definition (in the high level interface). This means that we cannot
		// (although in theory we could) pass placeholders to the computation which are not
		//also referenced in the functor.
		BOOST_STATIC_ASSERT(boost::mpl::size<typename derived_traits_t::template traits<Index>::local_domain_t::esf_args>::value==boost::mpl::size<typename derived_traits_t::template traits<Index>::functor_t::arg_list>::value);

                exec_functor_t::template execute_kernel< typename derived_traits_t::template traits<Index> >(local_domain, static_cast<const derived_t*>(this));

            }
        };

        /**
           \brief defines a method which associates an host_tmp_storage, whose range depends on an index, to the element in the Temporaries vector at that index position.
           \tparam Temporaries is the vector of temporary placeholder types.
        */
        template <typename Temporaries, typename Ranges, typename ValueType, typename LayoutType, uint_t BI, uint_t BJ, typename StrategyTraits, enumtype::backend BackendID>
        struct get_storage_type {
            template <typename Index>
            struct apply {
                typedef typename boost::mpl::at<Ranges, Index>::type range_type;

                typedef pair<typename StrategyTraits::template tmp<BackendID, ValueType, LayoutType, BI, BJ, -range_type::iminus::value, -range_type::jminus::value, range_type::iplus::value, range_type::jplus::value>::host_storage_t, typename boost::mpl::at<Temporaries, Index>::type::index_type> type;
            };
        };

/** metafunction to check whether the storage_type inside the PlcArgType is temporary */
        template <typename PlcArgType>
        struct is_temporary_arg : is_temporary_storage<typename PlcArgType::storage_type>
        {};
    }//namespace _impl



/** this struct contains the 'run' method for all backends, with a policy determining the specific type. Each backend contains a traits class for the specific case. */
    template< enumtype::backend BackendType, enumtype::strategy StrategyType >
    struct backend: public heap_allocated_temps<backend<BackendType, StrategyType > >
    {
        typedef backend_from_id <BackendType> backend_traits_t;
        typedef strategy_from_id <StrategyType> strategy_traits_t;
        static const enumtype::strategy s_strategy_id=StrategyType;
        static const enumtype::backend s_backend_id =BackendType;

        template <typename ValueType, typename Layout>
        struct storage_type {
            typedef typename backend_traits_t::template storage_traits<ValueType, Layout>::storage_t type;
        };


        template <typename ValueType, typename Layout>
        struct temporary_storage_type
            {
                /** temporary storage must have the same iterator type than the regular storage
                 */
            private:
                typedef typename backend_traits_t::template storage_traits<ValueType, Layout, true>::storage_t temp_storage_t;
            public:
                typedef typename boost::mpl::if_<typename boost::mpl::bool_<s_strategy_id==enumtype::Naive>::type,
                                       temp_storage_t,
                                       no_storage_type_yet< temp_storage_t > >::type type;
        };


/**

 */
        template <typename Domain
                  , typename MssType
                  , typename RangeSizes
                  , typename ValueType
                  , typename LayoutType >
        struct obtain_storage_types {

            static const uint_t tileI = (strategy_traits_t::BI);

            static const uint_t tileJ = (strategy_traits_t::BJ);

            typedef typename boost::mpl::fold<typename Domain::placeholders,
                boost::mpl::vector<>,
                boost::mpl::if_<
                    is_plchldr_to_temp<boost::mpl::_2>,
                    boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2 >,
                    boost::mpl::_1>
            >::type list_of_temporaries;

            typedef typename MssType::written_temps_per_functor written_temps_per_functor;

            typedef typename boost::mpl::transform<
                list_of_temporaries,
                _impl::associate_ranges<written_temps_per_functor, RangeSizes>
            >::type list_of_ranges;

            typedef boost::mpl::filter_view<typename Domain::placeholders, _impl::is_temporary_arg<boost::mpl::_> > temporaries;

            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<temporaries>::type::value> iter_range;

            typedef typename boost::mpl::fold<
                iter_range,
                boost::mpl::vector<>,
                typename boost::mpl::push_back<
                    typename boost::mpl::_1,
                    typename _impl::get_storage_type<
                        temporaries,
                        list_of_ranges,
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
         * \tparam FunctorList  List of functors to execute (in order)
         * \tparam range_sizes computed range sizes to know where to compute functot at<i>
         * \tparam LoopIntervals List of intervals on which functors are defined
         * \tparam FunctorsMap Map between interval and actual arguments to pass to Do methods
         * \tparam Domain Domain class (not really useful maybe)
         * \tparam Coords Coordinate class with domain sizes and splitter coordinates
         * \tparam LocalDomainList List of local domain to be pbassed to functor at<i>
         */
        template <typename FunctorList, // List of functors to execute (in order)
                  typename range_sizes, // computed range sizes to know where to compute functot at<i>
                  typename LoopIntervals, // List of intervals on which functors are defined
                  typename FunctorsMap,  // Map between interval and actual arguments to pass to Do methods
                  typename ExecutionEngine,
                  //typename Domain, // Domain class (not really useful maybe)
                  typename Coords, // Coordinate class with domain sizes and splitter coordinates
                  typename LocalDomainList
                  > // List of local domain to be pbassed to functor at<i>
        static void run(/*Domain const& domain, */Coords const& coords, LocalDomainList &local_domain_list) {// TODO: I would swap the arguments coords and local_domain_list here, for consistency
            //wrapping all the template arguments in a single container
            typedef typename boost::mpl::if_<typename boost::mpl::bool_< ExecutionEngine::type::iteration==enumtype::forward >::type, LoopIntervals, typename boost::mpl::reverse<LoopIntervals>::type >::type oriented_loop_intervals_t;

/**
   @brief template arguments container
   the only purpose of this struct is to collect template arguments in one single types container, in order to lighten the notation
*/
            /* struct arguments */
            /* { */
            /*     typedef FunctorList functor_list_t; */
            /*     typedef oriented_loop_intervals_t loop_intervals_t; */
            /*     typedef FunctorsMap functors_map_t; */
            /*     typedef range_sizes range_sizes_t; */
            /*     typedef LocalDomainList domain_list_t; */
            /*     typedef Coords coords_t; */
            /*     typedef ExecutionEngine execution_type_t; */
            /* }; */
            //Definition of a local struct to be passed as template parameter is a C++11 feature not supported by CUDA for __global__ functions

	    typedef arguments<FunctorList, oriented_loop_intervals_t, FunctorsMap, range_sizes, LocalDomainList, Coords, ExecutionEngine> args;

        typedef typename backend_traits_t::template execute_traits< args >::backend_t backend_t;
        strategy_from_id< s_strategy_id >::template loop< backend_t >::run_loop(local_domain_list, coords);
        }


        template <typename ArgList, typename Coords>
        static void prepare_temporaries(ArgList & arg_list, Coords const& coords)
            {
                _impl::template prepare_temporaries_functor<ArgList, Coords, s_strategy_id>::prepare_temporaries(/*std::forward<ArgList&>*/(arg_list), /*std::forward<Coords const&>*/(coords));
            }
    };


} // namespace gridtools
