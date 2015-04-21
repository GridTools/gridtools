#pragma once
#include <gt_for_each/for_each.hpp>
#include "../backend_traits_fwd.h"

/**@file
@brief type definitions and structures specific for the Host backend*/

namespace gridtools{
    namespace _impl_host{
        /**forward declaration*/
        template <typename Arguments>
        struct run_functor_host;
    }


    /**forward declaration*/
    template<typename T>
    struct wrap_pointer;

    /**Traits struct, containing the types which are specific for the host backend*/
    template<>
    struct backend_traits_from_id<enumtype::Host>{

        template <typename T>
        struct pointer
        {
            typedef wrap_pointer<T> type;
        };

        template <typename ValueType, typename Layout, bool Temp=false, short_t SpaceDim=1>
        struct storage_traits{
            typedef storage<base_storage<typename pointer<ValueType>::type, Layout, Temp, SpaceDim > >   storage_t;
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
        static uint_t processing_element_i() {
            return thread_id();
        }

        /** This is the function used by the specific backend
         *  that determines the j coordinate of a processing element.
         *  In the case of the host, a processing element is equivalent to an OpenMP core
         */
        static uint_t processing_element_j() {
            return 0;
        }

        //function alias (pre C++11, std::bind or std::mem_fn,
        //using function pointers looks very ugly)
        template<
            typename Sequence
            , typename F
            >
        //unnecessary copies/indirections if the compiler is not smart (std::forward)
        inline static void for_each(F f){
                gridtools::for_each<Sequence>(f);
            }

        template <uint_t Id>
        struct once_per_block {
            template<typename Left, typename Right>
            GT_FUNCTION//inline
            static void assign(Left& l, Right const& r){
                l=r;
            }
        };

        template<typename T> struct printu{BOOST_MPL_ASSERT_MSG((false), UUUUUUUU, (T));};
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
            template<typename LocalDomainList, typename Coords>
            static void run(LocalDomainList& local_domain_list, const Coords& coords, const uint_t bi, const uint_t bj)
            {
                //each strategy executes a different high level loop for a mss
                strategy_from_id<StrategyId>::template mss_loop<RunFunctorArgs, enumtype::Host>::template run(local_domain_list, coords, bi, bj);
            }
        };

        template<typename MssLocalDomain>
        struct fuse_mss_local_domain_strategy
        {
            BOOST_STATIC_ASSERT((is_mss_local_domain<MssLocalDomain>::value));
            typedef MssLocalDomain type;
        };

        template<typename MssLocalDomain, typename MergedMssLocalDomain>
        struct args_lookup_map
        {
            BOOST_STATIC_ASSERT((is_mss_local_domain<MssLocalDomain>::value));
            BOOST_STATIC_ASSERT((is_mss_local_domain<MergedMssLocalDomain>::value));
            typedef create_trivial_args_lookup_map<MssLocalDomain> type;
        };


    };

}//namespace gridtools
