#pragma once
#include "stencil-composition/backend_host/iterate_domain_host.hpp"

namespace gridtools {

    /**
 * @brief main functor that setups the CUDA kernel for a MSS and launchs it
 * @tparam RunFunctorArguments run functor argument type with the main configuration of the MSS
 */
    template <typename RunFunctorArguments >
    struct execute_kernel_functor_host
    {
        GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), "Internal Error: wrong type");
        typedef typename RunFunctorArguments::local_domain_t local_domain_t;
        typedef typename RunFunctorArguments::coords_t coords_t;

        /**
       @brief core of the kernel execution
       @tparam Traits traits class defined in \ref gridtools::_impl::run_functor_traits
    */
        explicit execute_kernel_functor_host(const local_domain_t& local_domain, const coords_t& coords,
                                             const uint_t first_i, const uint_t first_j, const uint_t loop_size_i, const uint_t loop_size_j,
                                             const uint_t block_idx_i, const uint_t block_idx_j)
            : m_local_domain(local_domain)
            , m_coords(coords)
            , m_first_pos(first_i, first_j)
            , m_loop_size(loop_size_i, loop_size_j)
            , m_block_id(block_idx_i, block_idx_j)
        {}

        // Naive strategy
        explicit  execute_kernel_functor_host(const local_domain_t& local_domain, const coords_t& coords)
            : m_local_domain(local_domain)
            , m_coords(coords)
            , m_first_pos(coords.i_low_bound(), coords.j_low_bound())
            //TODO strictling speaking the loop the size is with +1. Recompute the numbers here to be consistent
            //with the convention, but that require adapint also the rectangular grids
            , m_loop_size(coords.i_high_bound()-coords.i_low_bound(), coords.j_high_bound()-coords.j_low_bound())
            , m_block_id(0, 0) {}

        void operator()()
        {
            typedef typename RunFunctorArguments::loop_intervals_t loop_intervals_t;
            typedef typename RunFunctorArguments::execution_type_t execution_type_t;
            using grid_t = typename coords_t::grid_t;

            // in the host backend there should be only one esf per mss
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<typename RunFunctorArguments::range_sizes_t>::value==1),
                                    "Internal Error: wrong size");
            typedef typename boost::mpl::back<typename RunFunctorArguments::range_sizes_t>::type range_t;
            GRIDTOOLS_STATIC_ASSERT((is_range<range_t>::value), "Internal Error: wrong type");

            typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;
            typedef backend_traits_from_id<enumtype::Host> backend_traits_t;
            //#ifdef __VERBOSE__
            //        #pragma omp critical
            //        {
            //TODOCOSUNA Ranges in other grid have to become radius
            std::cout << "I loop " << m_first_pos[0] <<"+"<< range_t::iminus::value << " -> "
                      << m_first_pos[0] <<"+"<< m_loop_size[0] <<"+"<< range_t::iplus::value << "\n";
            std::cout << "J loop " << m_first_pos[1] <<"+"<< range_t::jminus::value << " -> "
                      << m_first_pos[1] <<"+"<< m_loop_size[1] <<"+"<< range_t::jplus::value << "\n";
            //        std::cout<<"iminus::value: "<<range_t::iminus::value<<std::endl;
            //        std::cout<<"iplus::value: "<<range_t::iplus::value<<std::endl;
            //        std::cout<<"jminus::value: "<<range_t::jminus::value<<std::endl;
            //        std::cout<<"jplus::value: "<<range_t::jplus::value<<std::endl;
            //        std::cout<<"block_id_i: "<<m_block_id[0]<<std::endl;
            //        std::cout<<"block_id_j: "<<m_block_id[1]<<std::endl;
            //        }
            //#endif

            typename iterate_domain_t::data_pointer_array_t data_pointer;
            typedef typename iterate_domain_t::strides_cached_t strides_t;
            strides_t strides;

            iterate_domain_t it_domain(m_local_domain, m_coords.grid());

            it_domain.set_data_pointer(&data_pointer);
            it_domain.set_strides_pointer(&strides);

            it_domain.template assign_storage_pointers<backend_traits_t >();
            it_domain.template assign_stride_pointers <backend_traits_t, strides_t>();


            typedef typename boost::mpl::front<loop_intervals_t>::type interval;
            typedef typename index_to_level<typename interval::first>::type from;
            typedef typename index_to_level<typename interval::second>::type to;
            typedef _impl::iteration_policy<from, to, zdim_index_t::value, execution_type_t::type::iteration> iteration_policy_t;

            //reset the index
            it_domain.set_index(0);

            it_domain.template initialize<0>(m_first_pos[0] + range_t::iminus::value, m_block_id[0]);
            //initialize color dim

            it_domain.template initialize<1>(0);
            it_domain.template initialize<2>(m_first_pos[1] + range_t::jminus::value, m_block_id[1]);
            it_domain.template initialize<3>( m_coords.template value_at< typename iteration_policy_t::from >() );


            typedef array<int_t, iterate_domain_t::N_META_STORAGES> array_index_t;
            typedef array<uint_t, 4> array_position_t;

            array_index_t memorized_index;
            array_position_t memorized_position;
            for(uint_t i=m_first_pos[0]; i <= m_first_pos[0] + m_loop_size[0];++i)
            {
                //TODO this n_colors is used by execute_kernel_functor_host.hpp, but because at the moment
                // it is not aware of the location type of iteration. In the future it should be extracted from cells or edges, etc.
                for(uint_t c=0; c < grid_t::n_colors; ++c)
                {
                    for(uint_t j=m_first_pos[1]; j <= m_first_pos[1] + m_loop_size[1];++j)
                    {
                        it_domain.get_index(memorized_index);
                        it_domain.get_position(memorized_position);

                        gridtools::for_each< loop_intervals_t >
                                ( _impl::run_f_on_interval<execution_type_t, RunFunctorArguments> (it_domain, m_coords) );
                        it_domain.set_index(memorized_index);
                        it_domain.set_position(memorized_position);
                        it_domain.template increment<2, static_int<1> >();
                    }
                    it_domain.template increment<2>( -(m_loop_size[1]+1));
                    it_domain.template increment<1, static_int<1> >();
                }
                it_domain.template increment<1, static_int<-grid_t::n_colors>>();
                it_domain.template increment<0,static_int<1> >();
            }
            it_domain.template increment<0>( -(m_loop_size[0]+1));
        }
    private:
        const local_domain_t& m_local_domain;
        const coords_t& m_coords;
        const gridtools::array<const uint_t, 2> m_first_pos;
        const gridtools::array<const uint_t, 2> m_loop_size;
        const gridtools::array<const uint_t, 2> m_block_id;
    };

}
