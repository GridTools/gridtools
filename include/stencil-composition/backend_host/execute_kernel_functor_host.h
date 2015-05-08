/*
 * execute_kernel_functor_host.h
 *
 *  Created on: Apr 25, 2015
 *      Author: cosuna
 */

#pragma once
#include "iterate_domain_host.h"
#include "../loop_hierarchy.h"

namespace gridtools {

/** @brief Partial specialization: naive and block implementation for the host backend */
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
            m_it_domain.template set_k_start( m_coords.template value_at< typename IterationPolicy::from >() );

            gridtools::for_each< LoopIntervals >
            ( RunOnInterval (m_it_domain, m_coords) );
        }
    };

    BOOST_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value));
    typedef typename RunFunctorArguments::local_domain_t local_domain_t;
    typedef typename RunFunctorArguments::coords_t coords_t;

    /**
       @brief core of the kernel execution
       \tparam Traits traits class defined in \ref gridtools::_impl::run_functor_traits
    */
    explicit execute_kernel_functor_host(const local_domain_t& local_domain, const coords_t& coords,
            const uint_t starti, const uint_t startj, const uint_t block_size_i, const uint_t block_size_j,
            const uint_t block_idx_i, const uint_t block_idx_j)
    : m_local_domain(local_domain)
    , m_coords(coords)
    , m_start(starti, startj)
    , m_block_size(block_size_i, block_size_j)
    , m_block_id(block_idx_i, block_idx_j)
    {}

    // Naive strategy
    explicit  execute_kernel_functor_host(const local_domain_t& local_domain, const coords_t& coords)
        : m_local_domain(local_domain)
    , m_coords(coords)
    , m_start(coords.i_low_bound(), coords.j_low_bound())
    , m_block_size(coords.i_high_bound()-coords.i_low_bound(), coords.j_high_bound()-coords.j_low_bound())
    , m_block_id(0, 0)

    {}

    void operator()()
    {
        typedef typename RunFunctorArguments::loop_intervals_t loop_intervals_t;
        typedef typename RunFunctorArguments::execution_type_t execution_type_t;

        // in the host backend there should be only one esf per mss
        BOOST_STATIC_ASSERT((boost::mpl::size<typename RunFunctorArguments::range_sizes_t>::value==1));
        typedef typename boost::mpl::back<typename RunFunctorArguments::range_sizes_t>::type range_t;
        BOOST_STATIC_ASSERT((is_range<range_t>::value));

        typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;
        typedef backend_traits_from_id<enumtype::Host> backend_traits_t;
#ifndef NDEBUG
        std::cout << "I loop " << m_start[0] <<"+"<< range_t::iminus::value << " -> "
                  << m_start[0] <<"+"<< m_block_size[0] <<"+"<< range_t::iplus::value << "\n";
        std::cout << "J loop " << m_start[1] <<"+"<< range_t::jminus::value << " -> "
                  << m_start[1] <<"+"<< m_block_size[1] <<"+"<< range_t::jplus::value << "\n";
        std::cout<<"iminus::value: "<<range_t::iminus::value<<std::endl;
#endif

        array<void* RESTRICT,iterate_domain_t::N_DATA_POINTERS> data_pointer;
        strides_cached<iterate_domain_t::N_STORAGES-1, typename local_domain_t::esf_args> strides;

        iterate_domain_t it_domain(m_local_domain);
        it_domain.template assign_storage_pointers<backend_traits_t >(&data_pointer);

        it_domain.template assign_stride_pointers <backend_traits_from_id<enumtype::Host> >(&strides);

        typedef typename boost::mpl::front<loop_intervals_t>::type interval;
        typedef typename index_to_level<typename interval::first>::type from;
        typedef typename index_to_level<typename interval::second>::type to;
        typedef _impl::iteration_policy<from, to, execution_type_t::type::iteration> iteration_policy;

        typedef array<uint_t, iterate_domain_t::N_STORAGES> array_t;
        loop_hierarchy<
            array_t, loop_item<0, enumtype::forward, int_t>,
            loop_item<1, enumtype::forward, int_t>
        > ij_loop(
                (int_t) (m_start[0] + range_t::iminus::value),
                (int_t) (m_start[0] + m_block_size[0] + range_t::iplus::value),
                (int_t) (m_start[1] + range_t::jminus::value),
                (int_t) (m_start[1] + m_block_size[1] + range_t::jplus::value)
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


//        void* data_pointer[iterate_domain_t::N_DATA_POINTERS];
//        iterate_domain_t it_domain(m_local_domain);
//
//        it_domain.template assign_storage_pointers<backend_traits_t >(data_pointer);
//
//        for (int_t i = m_starti + range_t::iminus::value;
//             i <= m_starti + m_block_size_i + range_t::iplus::value; ++i)
//        {
//            for (int_t j = m_startj + range_t::jminus::value;
//                j <= m_startj + m_block_size_j + range_t::jplus::value; ++j)
//            {
//                //reset the index
//                it_domain.set_index(0);
//                it_domain.template assign_ij<0>(i, m_block_idx_i);
//                it_domain.template assign_ij<1>(j, m_block_idx_j);
//                /** setting an iterator to the address of the current i,j entry to be accessed */
//                typedef typename boost::mpl::front<loop_intervals_t>::type interval;
//                typedef typename index_to_level<typename interval::first>::type from;
//                typedef typename index_to_level<typename interval::second>::type to;
//                typedef _impl::iteration_policy<from, to, execution_type_t::type::iteration> iteration_policy;
//                assert(i>=0);
//                assert(j>=0);
//
//                //setting the initial k level (for backward/parallel iterations it is not 0)
//                it_domain.set_k_start( m_coords.template value_at< typename iteration_policy::from >() );
//
//            //local structs can be passed as template arguments in C++11 (would improve readability)
//
//                /** run the iteration on the k dimension */
//                gridtools::for_each< loop_intervals_t > (
//                    _impl::run_f_on_interval<
//                            execution_type_t, RunFunctorArguments
//                         >(it_domain, m_coords) );
//
//            }
//        }
    }
private:
    const local_domain_t& m_local_domain;
    const coords_t& m_coords;
    gridtools::array<const uint_t, 2> m_start;
    gridtools::array<const uint_t, 2> m_block_size;
    gridtools::array<const uint_t, 2> m_block_id;

//    const uint_t m_starti;
//    const uint_t m_startj;
//    const uint_t m_block_size_i;
//    const uint_t m_block_size_j;
//    const uint_t m_block_idx_i;
//    const uint_t m_block_idx_j;
};

}
