#pragma once
#include <boost/mpl/for_each.hpp>

#include "../backend_traits_fwd.hpp"
#include "run_esf_functor_host.hpp"
#include "../block_size.hpp"
#include "iterate_domain_host.hpp"
#include "strategy_host.hpp"
#include "empty_iterate_domain_cache.hpp"

#ifdef ENABLE_METERS
#include "timer_host.hpp"
#else
#include "../timer_dummy.hpp"
#endif

/**@file
@brief type definitions and structures specific for the Host backend
*/
namespace gridtools {
    namespace _impl_host {
        /**forward declaration*/
        template < typename Arguments >
        struct run_functor_host;
    }

    /**forward declaration*/
    template < typename T, bool Array >
    struct wrap_pointer;

    /**Traits struct, containing the types which are specific for the host backend*/
    template <>
    struct backend_traits_from_id< enumtype::Host > {

        /**
           @brief pointer type associated to the host backend
         */
        template < typename T >
        struct pointer {
            typedef wrap_pointer< T > type;
        };

        /**
           @brief storage type associated to the host backend
         */
        template < typename ValueType, typename MetaData, bool Temp, short_t FieldDim = 1 >
        struct storage_traits {
            GRIDTOOLS_STATIC_ASSERT((is_meta_storage< MetaData >::value), "wrong type for the storage_info");
            typedef storage< base_storage< typename pointer< ValueType >::type, MetaData, FieldDim > > storage_t;
        };

        struct default_alignment {
            typedef aligned< 0 > type;
        };

        /**
           @brief storage info type associated to the host backend

           the storage info type is meta_storage_base, which is not clonable to GPU.
         */
        template < typename IndexType, typename Layout, bool Temp, typename Halo, typename Alignment >
        struct meta_storage_traits {
            GRIDTOOLS_STATIC_ASSERT((is_layout_map< Layout >::value), "wrong type for the storage_info");
            GRIDTOOLS_STATIC_ASSERT(is_halo< Halo >::type::value, "wrong type");
            GRIDTOOLS_STATIC_ASSERT(is_aligned< Alignment >::type::value, "wrong type");

            typedef meta_storage<
                meta_storage_aligned< meta_storage_base< IndexType::value, Layout, Temp >, Alignment, Halo > > type;
        };

        template < typename Arguments >
        struct execute_traits {
            typedef _impl_host::run_functor_host< Arguments > run_functor_t;
        };

        /** This is the function used by the specific backend to inform the
            generic backend and the temporary storage allocator how to
            compute the number of threads in the i-direction, in a 2D
            grid of threads.
        */
        static uint_t n_i_pes(uint_t = 0) {
#ifdef _OPENMP
            return omp_get_max_threads();
#else
            return 1;
#endif
        }

        /** This is the function used by the specific backend to inform the
            generic backend and the temporary storage allocator how to
            compute the number of threads in the j-direction, in a 2D
            grid of threads.
        */
        static uint_t n_j_pes(uint_t = 0) { return 1; }

        /** This is the function used by the specific backend
         *  that determines the i coordinate of a processing element.
         *  In the case of the host, a processing element is equivalent to an OpenMP core
         */
        static uint_t processing_element_i() {
#ifdef _OPENMP
            return omp_get_thread_num();
#else
            return 0;
#endif
        }

        /** This is the function used by the specific backend
         *  that determines the j coordinate of a processing element.
         *  In the case of the host, a processing element is equivalent to an OpenMP core
         */
        static uint_t processing_element_j() { return 0; }

        template < uint_t Id, typename BlockSize >
        struct once_per_block {
            GRIDTOOLS_STATIC_ASSERT((is_block_size< BlockSize >::value), "Error: wrong type");

            template < typename Left, typename Right >
            GT_FUNCTION // inline
                static void
                assign(Left &l, Right const &r) {
                l = (Left)r;
            }
        };

        /**
         * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         */
        template < typename RunFunctorArgs >
        struct mss_loop {
            typedef typename RunFunctorArgs::backend_ids_t backend_ids_t;

            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArgs >::value), "Internal Error: wrong type");
            template < typename LocalDomain, typename Grid, typename ReductionData >
            static void run(LocalDomain &local_domain,
                const Grid &grid,
                ReductionData &reduction_data,
                const uint_t bi,
                const uint_t bj) {
                GRIDTOOLS_STATIC_ASSERT((is_local_domain< LocalDomain >::value), "Internal Error: wrong type");
                GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "Internal Error: wrong type");
                GRIDTOOLS_STATIC_ASSERT((is_reduction_data< ReductionData >::value), "Internal Error: wrong type");

                // each strategy executes a different high level loop for a mss
                strategy_from_id_host< backend_ids_t::s_strategy_id >::template mss_loop<
                    RunFunctorArgs >::template run(local_domain, grid, reduction_data, bi, bj);
            }
        };

        /**
         * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
         */
        struct mss_fuse_esfs_strategy {
            typedef boost::mpl::bool_< false > type;
            BOOST_STATIC_CONSTANT(bool, value = (type::value));
        };

        // high level metafunction that contains the run_esf_functor corresponding to this backend
        typedef boost::mpl::quote2< run_esf_functor_host > run_esf_functor_h_t;

        // metafunction that contains the strategy from id metafunction corresponding to this backend
        template < typename BackendIds >
        struct select_strategy {
            GRIDTOOLS_STATIC_ASSERT((is_backend_ids< BackendIds >::value), "Error");
            typedef strategy_from_id_host< BackendIds::s_strategy_id > type;
        };

        /*
         * @brief metafunction that determines whether this backend requires redundant computations at halo points
         * of each block, given the strategy Id
         * @tparam StrategyId the strategy id
         * @return always false for Host
         */
        template < enumtype::strategy StrategyId >
        struct requires_temporary_redundant_halos {
            typedef
                typename boost::mpl::if_c< StrategyId == enumtype::Naive, boost::mpl::false_, boost::mpl::true_ >::type
                    type;
        };

        template < enumtype::strategy StrategyId >
        struct get_block_size {
            typedef typename strategy_from_id_host< StrategyId >::block_size_t type;
        };

        /**
         * @brief metafunction that returns the right iterate domain
         * (depending on whether the local domain is positional or not)
         * @param IterateDomainArguments the iterate domain arguments
         * @return the iterate domain type for this backend
         */
        template < typename IterateDomainArguments >
        struct select_iterate_domain {
            GRIDTOOLS_STATIC_ASSERT(
                (is_iterate_domain_arguments< IterateDomainArguments >::value), "Internal Error: wrong type");
// indirection in order to avoid instantiation of both types of the eval_if
#ifdef STRUCTURED_GRIDS
            template < typename _IterateDomainArguments >
            struct select_positional_iterate_domain {
                typedef iterate_domain_host< positional_iterate_domain, _IterateDomainArguments > type;
            };
#endif

            template < typename _IterateDomainArguments >
            struct select_basic_iterate_domain {
                typedef iterate_domain_host< iterate_domain, _IterateDomainArguments > type;
            };

            typedef typename boost::mpl::eval_if<
                local_domain_is_stateful< typename IterateDomainArguments::local_domain_t >,
#ifdef STRUCTURED_GRIDS
                select_positional_iterate_domain< IterateDomainArguments >,
#else
                select_basic_iterate_domain< IterateDomainArguments >,
#endif
                select_basic_iterate_domain< IterateDomainArguments > >::type type;
        };

        template < typename IterateDomainArguments >
        struct select_iterate_domain_cache {
            typedef empty_iterate_domain_cache type;
        };

#ifdef ENABLE_METERS
        typedef timer_host performance_meter_t;
#else
        typedef timer_dummy performance_meter_t;
#endif
    };

} // namespace gridtools
