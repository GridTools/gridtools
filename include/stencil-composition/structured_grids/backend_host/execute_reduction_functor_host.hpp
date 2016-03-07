#pragma once
#include "stencil-composition/backend_host/iterate_domain_host.hpp"
#include "stencil-composition/loop_hierarchy.hpp"
#include "../../iteration_policy.hpp"
#include "../../execution_policy.hpp"

namespace gridtools {

    namespace strgrid {

        template<typename T> struct printg{BOOST_MPL_ASSERT_MSG((false), GGGGGGGGGGGGGGGGGG, (T));};
        /**
        * @brief main functor that setups the CUDA kernel for a MSS and launchs it
        * @tparam RunFunctorArguments run functor argument type with the main configuration of the MSS
        */
        template <typename RunFunctorArguments >
        struct execute_reduction_functor_host
        {
            /**
            * @brief functor implementing the kernel executed in the innermost loop
            * This functor contains the portion of the code executed in the innermost loop. In this case it
            * is the loop over the third dimension (k), but the generality of the loop hierarchy implementation
            * allows to easily generalize this.
            */
            template<typename LoopIntervals, typename RunOnInterval, typename IterateDomain, typename Grid, typename  IterationPolicy>
            struct innermost_functor{

            private:

                IterateDomain & m_it_domain;
                const Grid& m_grid;

            public:

                IterateDomain const& it_domain() const { return m_it_domain; }

                innermost_functor(IterateDomain & it_domain, const Grid& grid):
                    m_it_domain(it_domain),
                    m_grid(grid){}

                void operator() () const {
                    m_it_domain.template initialize<2>( m_grid.template value_at< typename IterationPolicy::from >() );

                    boost::mpl::for_each< LoopIntervals >
                            ( RunOnInterval (m_it_domain, m_grid) );
                }
            };

            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), "Internal Error: wrong type");
            typedef typename RunFunctorArguments::local_domain_t local_domain_t;
            typedef typename RunFunctorArguments::grid_t grid_t;
            typedef typename RunFunctorArguments::functor_return_type_t reduced_value_t;

            /**
            * @brief core of the kernel execution
            * @tparam Traits traits class defined in \ref gridtools::_impl::run_functor_traits
            */
            explicit execute_reduction_functor_host(const local_domain_t& local_domain, const grid_t& grid,
                                                 const uint_t first_i, const uint_t first_j, const uint_t last_i, const uint_t last_j,
                                                 const uint_t block_idx_i, const uint_t block_idx_j)
                : m_local_domain(local_domain)
                , m_grid(grid)
#ifdef CXX11_ENABLED
                , m_first_pos{first_i, first_j}
                , m_loop_size{last_i, last_j}
                , m_block_id{block_idx_i, block_idx_j}
#else
                , m_first_pos(first_i, first_j)
                , m_loop_size(last_i, last_j)
                , m_block_id(block_idx_i, block_idx_j)
#endif
            {}

            // Naive strategy
            explicit  execute_reduction_functor_host(const local_domain_t& local_domain, const grid_t& grid)
                : m_local_domain(local_domain)
                , m_grid(grid)
#ifdef CXX11_ENABLED
                , m_first_pos{grid.i_low_bound(), grid.j_low_bound()}
                , m_loop_size{grid.i_high_bound()-grid.i_low_bound(), grid.j_high_bound()-grid.j_low_bound()}
                , m_block_id{0, 0}
#else
                , m_first_pos(grid.i_low_bound(), grid.j_low_bound())
                , m_loop_size(grid.i_high_bound()-grid.i_low_bound(), grid.j_high_bound()-grid.j_low_bound())
                , m_block_id(0, 0)
#endif
            {}

            void operator()()
            {
                typedef typename RunFunctorArguments::loop_intervals_t loop_intervals_t;
                typedef typename RunFunctorArguments::execution_type_t execution_type_t;

                // in the host backend there should be only one esf per mss
                GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<typename RunFunctorArguments::extent_sizes_t>::value==1),
                                        "Internal Error: wrong size");
                typedef typename boost::mpl::back<typename RunFunctorArguments::extent_sizes_t>::type extent_t;
                GRIDTOOLS_STATIC_ASSERT((is_extent<extent_t>::value), "Internal Error: wrong type");

                typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;
                typedef backend_traits_from_id<enumtype::Host> backend_traits_t;
#ifdef VERBOSE
#pragma omp critical
                {
                    std::cout << "I loop " << m_first_pos[0] <<"+"<< extent_t::iminus::value << " -> "
                              << m_first_pos[0] <<"+"<< m_loop_size[0] <<"+"<< extent_t::iplus::value << "\n";
                    std::cout << "J loop " << m_first_pos[1] <<"+"<< extent_t::jminus::value << " -> "
                              << m_first_pos[1] <<"+"<< m_loop_size[1] <<"+"<< extent_t::jplus::value << "\n";
                    std::cout<<"iminus::value: "<<extent_t::iminus::value<<std::endl;
                    std::cout<<"iplus::value: "<<extent_t::iplus::value<<std::endl;
                    std::cout<<"jminus::value: "<<extent_t::jminus::value<<std::endl;
                    std::cout<<"jplus::value: "<<extent_t::jplus::value<<std::endl;
                    std::cout<<"block_id_i: "<<m_block_id[0]<<std::endl;
                    std::cout<<"block_id_j: "<<m_block_id[1]<<std::endl;
                }
#endif

                typename iterate_domain_t::data_pointer_array_t data_pointer;
                typedef typename iterate_domain_t::strides_cached_t strides_t;
                strides_t strides;

                iterate_domain_t it_domain(m_local_domain);

                it_domain.set_data_pointer_impl(&data_pointer);
                it_domain.set_strides_pointer_impl(&strides);

                it_domain.template assign_storage_pointers<backend_traits_t >();
                it_domain.template assign_stride_pointers <backend_traits_t, strides_t>();

                typedef typename boost::mpl::front<loop_intervals_t>::type interval;
                typedef typename index_to_level<typename interval::first>::type from;
                typedef typename index_to_level<typename interval::second>::type to;
                typedef _impl::iteration_policy<from, to, zdim_index_t::value, execution_type_t::type::iteration> iteration_policy_t;

                reduced_value_t& reduced_value = it_domain.reduced_value();

                //reset the index
                it_domain.set_index(0);

                //TODO FUSING work on extending the loops using the extent
//                it_domain.template initialize<0>(m_first_pos[0] + extent_t::iminus::value, m_block_id[0]);
                it_domain.template initialize<0>(m_first_pos[0], m_block_id[0]);
                it_domain.template initialize<1>(m_first_pos[1], m_block_id[1]);
                it_domain.template initialize<2>( m_grid.template value_at< typename iteration_policy_t::from >() );


                typedef array<int_t, iterate_domain_t::N_META_STORAGES> array_index_t;
                typedef array<uint_t, 4> array_position_t;

                array_index_t memorized_index;

                assert(m_loop_size[1]>0);

                typedef  typename boost::mpl::first< typename boost::mpl::front<loop_intervals_t>::type>::type first_lev_index;
                typedef typename index_to_level<first_lev_index>::type first_lev;
                typedef level<first_lev::Splitter::value, first_lev::Offset::value + 1 > first_lev_plus1;


//                it_domain.get_index(memorized_index);
//                boost::mpl::for_each< loop_intervals_t >(
//                    _impl::run_f_on_interval< execution_type_t, RunFunctorArguments >(it_domain, m_grid));
//                it_domain.set_index(memorized_index);

                for (uint_t i = m_first_pos[0]; i <= m_first_pos[0] + m_loop_size[0]; ++i) {
                    it_domain.get_index(memorized_index);

                    boost::mpl::for_each< loop_intervals_t >(
                        _impl::run_f_on_interval< execution_type_t, RunFunctorArguments >(it_domain, m_grid));
                    it_domain.set_index(memorized_index);
                    it_domain.template increment<0, static_int< 1 > >();
                }
                it_domain.template increment<0>(-(m_loop_size[0] + 1));
                it_domain.template increment<1, static_int< 1 > >();


                for (uint_t j = m_first_pos[1]+1; j <= m_first_pos[1] + m_loop_size[1]; ++j) {

                    for (uint_t i = m_first_pos[0]; i <= m_first_pos[0] + m_loop_size[0]; ++i) {
                        it_domain.get_index(memorized_index);

                        boost::mpl::for_each< loop_intervals_t >(
                            _impl::run_f_on_interval< execution_type_t, RunFunctorArguments >(it_domain, m_grid));
                        it_domain.set_index(memorized_index);
                        it_domain.template increment<0, static_int< 1 > >();
                    }
                    it_domain.template increment<0>(-(m_loop_size[0] + 1));
                    it_domain.template increment<1, static_int< 1 > >();
                }
                it_domain.template increment<1>(-(m_loop_size[1] + 1));
            }

        private:
            const local_domain_t& m_local_domain;
            const grid_t& m_grid;
            const gridtools::array<const uint_t, 2> m_first_pos;
            const gridtools::array<const uint_t, 2> m_loop_size;
            const gridtools::array<const uint_t, 2> m_block_id;
        };
    } // namespace strgrid
} //namespace gridtools
