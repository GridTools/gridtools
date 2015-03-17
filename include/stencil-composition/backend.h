#pragma once

#include <boost/mpl/filter_view.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/reverse.hpp>

#include <gridtools.h>

#include "backend_traits_fwd.h"
#ifdef __CUDACC__
#include <stencil-composition/backend_cuda/backend_cuda.h>
#else
#include <stencil-composition/backend_host/backend_host.h>
#endif

#include "../common/pair.h"
#include "../common/gridtools_runtime.h"
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
           \brief defines a method which associates an host_tmp_storage, whose range depends on an index, to the element in the Temporaries vector at that index position.
           \tparam Temporaries is the vector of temporary placeholder types.
        */
        template <typename Temporaries,
                  typename Ranges,
                  typename ValueType,
                  typename LayoutType,
                  uint_t BI, uint_t BJ,
                  typename StrategyTraits,
                  enumtype::backend BackendID>
        struct get_storage_type {
            template <typename Index>
            struct apply {
                typedef typename boost::mpl::at<Ranges, Index>::type range_type;

                typedef pair<
                    typename StrategyTraits::template get_tmp_storage<
                        BackendID, 
                        ValueType, 
                        LayoutType, 
                        BI, BJ, 
                        -range_type::iminus::value, 
                        -range_type::jminus::value, 
                        range_type::iplus::value, 
                        range_type::jplus::value>::type, 
                    typename boost::mpl::at<Temporaries, Index>::type::index_type
                    > type;
            };
        };

        /** metafunction to check whether the storage_type inside the PlcArgType is temporary */
        template <typename PlcArgType>
        struct is_temporary_arg : is_temporary_storage<typename PlcArgType::storage_type>
        {};
    }//namespace _impl


	/** The following struct is defined here since the current version of NVCC does not accept local types to be used as template arguments of __global__ functions \todo move inside backend::run()*/
	template<typename FunctorList, typename LoopIntervals, typename FunctorsMap, typename RangeSizes, typename LocalDomainList, typename Coords, typename ExecutionEngine>
    struct arguments
    {
        typedef FunctorList functor_list_t;
        typedef LoopIntervals loop_intervals_t;
        typedef FunctorsMap functors_map_t;
        typedef RangeSizes range_sizes_t;
        typedef LocalDomainList domain_list_t;
        typedef Coords coords_t;
        typedef ExecutionEngine execution_type_t;
    };

    /**
       @brief traits struct for the run_functor
       Specialization for all backend classes.

       This struct defines a type for all the template arguments in
       the run_functor subclasses. It is required because in the
       run_functor class definition the 'Derived' template argument is
       an incomplete type (ans thus we can not access its template
       arguments).  This struct also contains all the type definitions
       common to all backends.
    */
    template <
        typename Arguments,
        template < typename Argument > class Back
        >
    struct run_functor_traits< Back< Arguments > >
    {
        typedef Arguments arguments_t;
        typedef typename Arguments::functor_list_t functor_list_t;
        typedef typename Arguments::loop_intervals_t loop_intervals_t;
        typedef typename Arguments::functors_map_t functors_map_t;
        typedef typename Arguments::range_sizes_t range_sizes_t;
        typedef typename Arguments::domain_list_t domain_list_t;
        typedef typename Arguments::coords_t coords_t;
        typedef Back<Arguments> backend_t;

        /**
           @brief traits class to be used inside the functor 
           \ref gridtools::_impl::execute_kernel_functor, which dependson an Index type.
        */
        template <typename Index>
        struct traits{
            typedef typename boost::mpl::at<range_sizes_t, Index>::type range_t;
            typedef typename boost::mpl::at<functor_list_t, Index>::type functor_t;
            typedef typename boost::fusion::result_of::value_at<domain_list_t, Index>::type local_domain_t;
            typedef typename boost::mpl::at<functors_map_t, Index>::type interval_map_t;
            typedef typename index_to_level<
                typename boost::mpl::deref<
                    typename boost::mpl::find_if<
                        loop_intervals_t,
                        boost::mpl::has_key<interval_map_t, boost::mpl::_1>
                        >::type
                    >::type::first
                >::type first_hit_t;

            typedef typename local_domain_t::iterate_domain_t iterate_domain_t;
        };
    };


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

        backend_from_id<type>

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
    template< enumtype::backend BackendType, enumtype::strategy StrategyType >
    struct backend
    {
        typedef backend_from_id <BackendType> backend_traits_t;
        typedef strategy_from_id <StrategyType> strategy_traits_t;
        typedef backend<BackendType, StrategyType> this_type;
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
            typedef typename boost::mpl::if_<typename boost::mpl::bool_< ExecutionEngine::type::iteration==enumtype::forward >::type, 
                LoopIntervals, 
                typename boost::mpl::reverse<LoopIntervals>::type >::type 
            oriented_loop_intervals_t;

            /**
               @brief template arguments container
               the only purpose of this struct is to collect template arguments in one single types container, in order to lighten the notation
            */
            typedef arguments<FunctorList, oriented_loop_intervals_t, FunctorsMap, range_sizes, LocalDomainList, Coords, ExecutionEngine> args;

            typedef typename backend_traits_t::template execute_traits< args >::backend_t backend_t;
            strategy_from_id< s_strategy_id >::template loop< backend_t >::run_loop(local_domain_list, coords);
        }


        template <typename ArgList, typename Coords>
        static void prepare_temporaries(ArgList & arg_list, Coords const& coords)
        {
            _impl::template prepare_temporaries_functor<ArgList, Coords, this_type>::
                prepare_temporaries((arg_list), (coords));
        }

        /** Initial interface

            Threads are oganized in a 2D grid. These two functions 
            n_i_threads() and n_j_threasd() retrieve the
            information about the sizes of this grid.

            n_i_threads() number of threads on the first dimension of the thread grid
        */
        static uint_t n_i_threads() {
            return n_threads();
        }

        /** Initial interface

            Threads are oganized in a 2D grid. These two functions 
            n_i_threads() and n_j_threasd() retrieve the
            information about the sizes of this grid.

            n_j_threads() number of threads on the second dimension of the thread grid
        */
        static uint_t n_j_threads() {
            return 1;
        }


    }; // struct backend {


} // namespace gridtools
