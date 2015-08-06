/*
 * execute_kernel_functor_host.h
 *
 *  Created on: Apr 25, 2015
 *      Author: cosuna
 */

#pragma once
#include "iterate_domain_host.hpp"
#include "../loop_hierarchy.hpp"

namespace gridtools {

/**
 * @brief main functor that setups the CUDA kernel for a MSS and launchs it
 * @tparam RunFunctorArguments run functor argument type with the main configuration of the MSS
 */
template <typename RunFunctorArguments >
struct execute_kernel_functor_host
{
    /**
       @brief functor implementing the kernel executed in the innermost loop

       This functor contains the portion of the code executed in the innermost loop. In this case it
       is the loop over the third dimension (k), but the generality of the loop hierarchy implementation
       allows to easily generalize this.
   */
    template<typename LoopIntervals, typename RunOnInterval, typename IterateDomain, typename Coords, typename  IterationPolicy>
    struct innermost_functor{

    private:

        IterateDomain & m_it_domain;
        const Coords& m_coords;

    public:

        IterateDomain const& it_domain() const { return m_it_domain; }

        innermost_functor(IterateDomain & it_domain, const Coords& coords):
            m_it_domain(it_domain),
            m_coords(coords){}

        void operator() () const {
            m_it_domain.template initialize<2>( m_coords.template value_at< typename IterationPolicy::from >() );

            gridtools::for_each< LoopIntervals >
            ( RunOnInterval (m_it_domain, m_coords) );
        }
    };

    GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), "Internal Error: wrong type");
    typedef typename RunFunctorArguments::local_domain_t local_domain_t;
    typedef typename RunFunctorArguments::coords_t coords_t;

    /**
       @brief core of the kernel execution
       @tparam Traits traits class defined in \ref gridtools::_impl::run_functor_traits
    */
    explicit execute_kernel_functor_host(const local_domain_t& local_domain, const coords_t& coords,
            const uint_t first_i, const uint_t first_j, const uint_t last_i, const uint_t last_j,
            const uint_t block_idx_i, const uint_t block_idx_j)
    : m_local_domain(local_domain)
    , m_coords(coords)
    , m_first_pos(first_i, first_j)
    , m_last_pos(last_i, last_j)
    , m_block_id(block_idx_i, block_idx_j)
    {}

    // Naive strategy
    explicit  execute_kernel_functor_host(const local_domain_t& local_domain, const coords_t& coords)
        : m_local_domain(local_domain)
    , m_coords(coords)
    , m_first_pos(coords.i_low_bound(), coords.j_low_bound())
    , m_last_pos(coords.i_high_bound()-coords.i_low_bound(), coords.j_high_bound()-coords.j_low_bound())
    , m_block_id(0, 0) {}

    void operator()()
    {
        typedef typename RunFunctorArguments::loop_intervals_t loop_intervals_t;
        typedef typename RunFunctorArguments::execution_type_t execution_type_t;

        // in the host backend there should be only one esf per mss
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<typename RunFunctorArguments::range_sizes_t>::value==1),
                                "Internal Error: wrong size");
        typedef typename boost::mpl::back<typename RunFunctorArguments::range_sizes_t>::type range_t;
        GRIDTOOLS_STATIC_ASSERT((is_range<range_t>::value), "Internal Error: wrong type");

        typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;
        typedef backend_traits_from_id<enumtype::Host> backend_traits_t;
#ifdef __VERBOSE__
        #pragma omp critical
        {
        std::cout << "I loop " << m_first_pos[0] <<"+"<< range_t::iminus::value << " -> "
                  << m_first_pos[0] <<"+"<< m_last_pos[0] <<"+"<< range_t::iplus::value << "\n";
        std::cout << "J loop " << m_first_pos[1] <<"+"<< range_t::jminus::value << " -> "
                  << m_first_pos[1] <<"+"<< m_last_pos[1] <<"+"<< range_t::jplus::value << "\n";
        std::cout<<"iminus::value: "<<range_t::iminus::value<<std::endl;
        std::cout<<"iplus::value: "<<range_t::iplus::value<<std::endl;
        std::cout<<"jminus::value: "<<range_t::jminus::value<<std::endl;
        std::cout<<"jplus::value: "<<range_t::jplus::value<<std::endl;
        std::cout<<"block_id_i: "<<m_block_id[0]<<std::endl;
        std::cout<<"block_id_j: "<<m_block_id[1]<<std::endl;
        }
#endif

        array<void* RESTRICT,iterate_domain_t::N_DATA_POINTERS> data_pointer;
        typedef strides_cached<iterate_domain_t::N_STORAGES-1, typename local_domain_t::esf_args> strides_t;
        strides_t strides;

        iterate_domain_t it_domain(m_local_domain);
        it_domain.set_data_pointer_impl(&data_pointer);
        it_domain.set_strides_pointer_impl(&strides);

        it_domain.template assign_storage_pointers<backend_traits_t >();
        it_domain.template assign_stride_pointers <backend_traits_t, strides_t>();

        typedef typename boost::mpl::front<loop_intervals_t>::type interval;
        typedef typename index_to_level<typename interval::first>::type from;
        typedef typename index_to_level<typename interval::second>::type to;
        typedef _impl::iteration_policy<from, to, execution_type_t::type::iteration> iteration_policy;

        typedef array<int_t, iterate_domain_t::N_STORAGES> array_t;
        loop_hierarchy<
            array_t, loop_item<0, int_t, 1>,
            loop_item<1, int_t, 1>
        > ij_loop(
                (int_t) (m_first_pos[0] + range_t::iminus::value),
                (int_t) (m_first_pos[0] + m_last_pos[0] + range_t::iplus::value),
                (int_t) (m_first_pos[1] + range_t::jminus::value),
                (int_t) (m_first_pos[1] + m_last_pos[1] + range_t::jplus::value)
        );

        //reset the index
        it_domain.set_index(0);
        ij_loop.initialize(it_domain, m_block_id);

        //define the kernel functor
        typedef innermost_functor<
                loop_intervals_t,
                _impl::run_f_on_interval<
                    execution_type_t,
                    RunFunctorArguments
                >,
                iterate_domain_t,
                coords_t,
                iteration_policy
        > innermost_functor_t;

        //instantiate the kernel functor
        innermost_functor_t f(it_domain, m_coords);

        //run the nested ij loop
        ij_loop.apply(it_domain, f);
    }
private:
    const local_domain_t& m_local_domain;
    const coords_t& m_coords;
    const gridtools::array<const uint_t, 2> m_first_pos;
    const gridtools::array<const uint_t, 2> m_last_pos;
    const gridtools::array<const uint_t, 2> m_block_id;
};

}
