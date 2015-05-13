#pragma once
#include <gt_for_each/for_each.hpp>
#include "../backend_traits_fwd.h"
#include "run_esf_functor_host.h"
#include "../block_size.h"
#include "iterate_domain_host.h"
#include "strategy_host.h"

/**@file
@brief type definitions and structures specific for the Host backend
*/
namespace gridtools{
    namespace _impl_host{
        /**forward declaration*/
        template <typename Arguments>
        struct run_functor_host;
    }

    namespace multithreading{
        /**
           Global variable storing the current thread id
        */
        int __attribute__((weak)) gt_thread_id;
#pragma omp threadprivate(gt_thread_id)

    }//namespace multithreading

    /**forward declaration*/
    template<typename T>
    struct wrap_pointer;

    /**Traits struct, containing the types which are specific for the host backend*/
    template<>
    struct backend_traits_from_id<enumtype::Host>{

        /**
           @brief pointer type associated to the host backend
         */
        template <typename T>
        struct pointer
        {
            typedef wrap_pointer<T> type;
        };

        /**
           @brief storage type associated to the host backend
         */
        template <typename ValueType, typename Layout, bool Temp=false, short_t FieldDim=1>
        struct storage_traits{
            typedef storage<base_storage<typename pointer<ValueType>::type, Layout, Temp, FieldDim > >   storage_t;
        };

        template <typename Arguments>
        struct execute_traits{
            typedef _impl_host::run_functor_host< Arguments > run_functor_t;
        };

        /** This is the function used by the specific backend to inform the
            generic backend and the temporary storage allocator how to
            compute the number of threads in the i-direction, in a 2D
            grid of threads.
        */
        static uint_t n_i_pes(int = 0) {
            return n_threads();
        }

        /** This is the function used by the specific backend to inform the
            generic backend and the temporary storage allocator how to
            compute the number of threads in the j-direction, in a 2D
            grid of threads.
        */
        static uint_t n_j_pes(int = 0) {
            return 1;
        }

        /** This is the function used by the specific backend
         *  that determines the i coordinate of a processing element.
         *  In the case of the host, a processing element is equivalent to an OpenMP core
         */
        static uint_t processing_element_i()  {
            return multithreading::gt_thread_id;
        }

        /**@brief set the thread id

           this method when openmp is enabled calls the thread_id() routine.
           NOTE: the reason why the latter routine is not called directly every time
           the thread id is queried is a possibly non-negligible overhead introduced by such call.
           This interface allows to store the thread id in a global variable when a parallel
           region is encountered.
         */
        static void set_thread_id(){
            multithreading::gt_thread_id = thread_id();
        }
        /** This is the function used by the specific backend
         *  that determines the j coordinate of a processing element.
         *  In the case of the host, a processing element is equivalent to an OpenMP core
         */
        static uint_t  processing_element_j()  {
            return 0;
        }

        //function alias (pre C++11, std::bind or std::mem_fn,
        //using function pointers looks very ugly)
        template<
            typename Sequence
            , typename F
            >

#ifdef CXX11_ENABLED
        //unnecessary copies/indirections if the compiler is not smart
        inline static void for_each(F&& f){
            gridtools::for_each<Sequence>(std::forward<F>(f));
            }
#else
        //unnecessary copies/indirections if the compiler is not smart
        inline static void for_each(F f){
            gridtools::for_each<Sequence>(f);
            }
#endif

        template <uint_t Id>
        struct once_per_block {
            template<typename Left, typename Right>
            GT_FUNCTION//inline
            static void assign(Left& l, Right const& r){
                l=r;
            }
        };

        /**
         * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         * @tparam StrategyId id of the strategy
         */
        template<typename RunFunctorArgs, enumtype::strategy StrategyId>
        struct mss_loop
        {
            BOOST_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value));
            template<typename LocalDomain, typename Coords>
            static void run(LocalDomain& local_domain, const Coords& coords, const uint_t bi, const uint_t bj)
            {
                BOOST_STATIC_ASSERT((is_local_domain<LocalDomain>::value));
                //each strategy executes a different high level loop for a mss
                strategy_from_id_host<StrategyId>::template mss_loop<RunFunctorArgs, enumtype::Host>::template run(local_domain, coords, bi, bj);
            }
        };

        /**
         * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
         */
        struct mss_fuse_esfs_strategy
        {
            typedef boost::mpl::bool_<false> type;
            BOOST_STATIC_CONSTANT(bool, value=(type::value));
        };

        // high level metafunction that contains the run_esf_functor corresponding to this backend
        typedef boost::mpl::quote2<run_esf_functor_host> run_esf_functor_h_t;

        // metafunction that contains the strategy from id metafunction corresponding to this backend
        template<enumtype::strategy StrategyId>
        struct select_strategy
        {
            typedef strategy_from_id_host<StrategyId> type;
        };

        template<enumtype::strategy StrategyId>
        struct requires_temporary_redundant_halos
        {
            typedef typename boost::mpl::if_c<
                StrategyId == enumtype::Naive,
                boost::mpl::false_,
                boost::mpl::true_
            >::type type;
        };

        template<enumtype::strategy StrategyId>
        struct get_block_size {
            typedef typename strategy_from_id_host<StrategyId>::block_size_t type;
        };

        /**
         * @brief metafunction that derives that returns the right iterate domain
         * (depending on whether the local domain is positional or not)
         * @param LocalDomain the local domain
         * @return the iterate domain type for this backend
         */
        template <typename LocalDomain>
        struct select_iterate_domain {
            BOOST_STATIC_ASSERT((is_local_domain<LocalDomain>::value));
            //indirection in order to avoid instantiation of both types of the eval_if
            template<typename _LocalDomain>
            struct select_positional_iterate_domain
            {
                typedef iterate_domain_host<positional_iterate_domain, _LocalDomain> type;
            };

            template<typename _LocalDomain>
            struct select_basic_iterate_domain
            {
                typedef iterate_domain_host<iterate_domain, _LocalDomain> type;
            };

            typedef typename boost::mpl::eval_if<
                local_domain_is_stateful<LocalDomain>,
                select_positional_iterate_domain<LocalDomain>,
                select_basic_iterate_domain<LocalDomain>
            >::type type;
        };
    };

}//namespace gridtools
