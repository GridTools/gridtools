#pragma once
#include "../../common/generic_metafunctions/variadic_to_vector.hpp"
#include "../../common/generic_metafunctions/transform_metadata.hpp"
#include "stencil-composition/backend_host/iterate_domain_host.hpp"
#include "stencil-composition/icosahedral_grids/esf_metafunctions.hpp"
#include "../../iteration_policy.hpp"
#include "../../execution_policy.hpp"
#include "../../grid_traits_fwd.hpp"

namespace gridtools {

    namespace icgrid {

        template < typename RunFunctorArguments, typename Index >
        struct colorize_run_functor_arguments {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), "Error");
            typedef typename transform_meta_data< RunFunctorArguments,
                typename RunFunctorArguments::color_t,
                color_type< (uint_t)Index::value > >::type type;
        };

        template < typename RunFunctorArguments, typename IterateDomain, typename Grid >
        struct color_execution_functor {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArguments >::value), "ERROR");
            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< IterateDomain >::value), "ERROR");
            GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "ERROR");

            typedef typename RunFunctorArguments::loop_intervals_t loop_intervals_t;
            typedef typename RunFunctorArguments::execution_type_t execution_type_t;

            typedef array< int_t, IterateDomain::N_META_STORAGES > array_index_t;
            typedef array< uint_t, 4 > array_position_t;

          private:
            IterateDomain &m_it_domain;
            Grid const &m_grid;
            gridtools::array< const uint_t, 2 > const &m_first_pos;
            gridtools::array< const uint_t, 2 > const &m_loop_size;
            const uint_t m_addon;

          public:
            color_execution_functor(IterateDomain &it_domain,
                Grid const &grid,
                gridtools::array< const uint_t, 2 > const &first_pos,
                gridtools::array< const uint_t, 2 > const &loop_size,
                const uint_t addon)
                : m_it_domain(it_domain), m_grid(grid), m_first_pos(first_pos), m_loop_size(loop_size), m_addon(addon) {
            }

            template < typename Index >
            void operator()(Index const &) const {

                array_index_t memorized_index;
                array_position_t memorized_position;

                for (uint_t j = m_first_pos[1]; j <= m_first_pos[1] + m_loop_size[1] + m_addon; ++j) {
                    m_it_domain.get_index(memorized_index);
                    m_it_domain.get_position(memorized_position);

                    // we fill the run_functor_arguments with the current color being processed
                    typedef typename colorize_run_functor_arguments< RunFunctorArguments, Index >::type
                        run_functor_arguments_t;
                    boost::mpl::for_each< loop_intervals_t >(
                        _impl::run_f_on_interval< execution_type_t, run_functor_arguments_t >(m_it_domain, m_grid));
                    m_it_domain.set_index(memorized_index);
                    m_it_domain.set_position(memorized_position);
                    m_it_domain.template increment< grid_traits_from_id< enumtype::icosahedral >::dim_j_t::value,
                        static_int< 1 > >();
                }
                m_it_domain.template increment< grid_traits_from_id< enumtype::icosahedral >::dim_j_t::value >(
                    -(m_loop_size[1] + 1 + m_addon));
                m_it_domain.template increment< grid_traits_from_id< enumtype::icosahedral >::dim_c_t::value,
                    static_int< 1 > >();
            }
        };
        /**
        * @brief main functor that setups the CUDA kernel for a MSS and launchs it
        * @tparam RunFunctorArguments run functor argument type with the main configuration of the MSS
        */
        template < typename RunFunctorArguments >
        struct execute_kernel_functor_host {
            GRIDTOOLS_STATIC_ASSERT(
                (is_run_functor_arguments< RunFunctorArguments >::value), "Internal Error: wrong type");
            typedef typename RunFunctorArguments::local_domain_t local_domain_t;
            typedef typename RunFunctorArguments::grid_t grid_t;
            typedef typename RunFunctorArguments::esf_sequence_t esf_sequence_t;
            typedef typename RunFunctorArguments::reduction_data_t reduction_data_t;

            typedef typename extract_esf_location_type< esf_sequence_t >::type location_type_t;

            using n_colors_t = typename location_type_t::n_colors;

            /**
            @brief core of the kernel execution
            @tparam Traits traits class defined in \ref gridtools::_impl::run_functor_traits
            */
            explicit execute_kernel_functor_host(const local_domain_t &local_domain,
                const grid_t &grid,
                reduction_data_t &reduction_data,
                const uint_t first_i,
                const uint_t first_j,
                const uint_t loop_size_i,
                const uint_t loop_size_j,
                const uint_t block_idx_i,
                const uint_t block_idx_j)
                : m_local_domain(local_domain), m_grid(grid), m_first_pos{first_i, first_j},
                  m_loop_size{loop_size_i, loop_size_j}, m_block_id{block_idx_i, block_idx_j} {}

            // Naive strategy
            explicit execute_kernel_functor_host(
                const local_domain_t &local_domain, const grid_t &grid, reduction_data_t &reduction_data)
                : m_local_domain(local_domain), m_grid(grid), m_first_pos{grid.i_low_bound(), grid.j_low_bound()}
                  // TODO strictling speaking the loop the size is with +1. Recompute the numbers here to be consistent
                  // with the convention, but that require adapint also the rectangular grids
                  ,
                  m_loop_size{grid.i_high_bound() - grid.i_low_bound(), grid.j_high_bound() - grid.j_low_bound()},
                  m_block_id{0, 0} {}

            void operator()() {
                typedef typename RunFunctorArguments::loop_intervals_t loop_intervals_t;
                typedef typename RunFunctorArguments::execution_type_t execution_type_t;
                using grid_topology_t = typename grid_t::grid_topology_t;

                // in the host backend there should be only one esf per mss
                GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< typename RunFunctorArguments::extent_sizes_t >::value == 1),
                    "Internal Error: wrong size");
                typedef typename boost::mpl::back< typename RunFunctorArguments::extent_sizes_t >::type extent_t;
                GRIDTOOLS_STATIC_ASSERT((is_extent< extent_t >::value), "Internal Error: wrong type");

                typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;
                typedef backend_traits_from_id< enumtype::Host > backend_traits_t;
                //#ifdef __VERBOSE__
                //        #pragma omp critical
                //        {
                //        std::cout<<"iminus::value: "<<extent_t::iminus::value<<std::endl;
                //        std::cout<<"iplus::value: "<<extent_t::iplus::value<<std::endl;
                //        std::cout<<"jminus::value: "<<extent_t::jminus::value<<std::endl;
                //        std::cout<<"jplus::value: "<<extent_t::jplus::value<<std::endl;
                //        std::cout<<"block_id_i: "<<m_block_id[0]<<std::endl;
                //        std::cout<<"block_id_j: "<<m_block_id[1]<<std::endl;
                //        }
                //#endif

                typename iterate_domain_t::data_pointer_array_t data_pointer;
                typedef typename iterate_domain_t::strides_cached_t strides_t;
                strides_t strides;

                iterate_domain_t it_domain(m_local_domain, m_grid.grid_topology());

                it_domain.set_data_pointer_impl(&data_pointer);
                it_domain.set_strides_pointer_impl(&strides);

                it_domain.template assign_storage_pointers< backend_traits_t >();
                it_domain.template assign_stride_pointers< backend_traits_t, strides_t >();

                typedef typename boost::mpl::front< loop_intervals_t >::type interval;
                typedef typename index_to_level< typename interval::first >::type from;
                typedef typename index_to_level< typename interval::second >::type to;
                typedef _impl::iteration_policy< from,
                    to,
                    typename grid_traits_from_id< enumtype::icosahedral >::dim_k_t,
                    execution_type_t::type::iteration > iteration_policy_t;

                // reset the index
                it_domain.set_index(0);

                // TODO FUSING work on extending the loops using the extent
                //                it_domain.template initialize<0>(m_first_pos[0] + extent_t::iminus::value,
                //                m_block_id[0]);
                it_domain.template initialize< grid_traits_from_id< enumtype::icosahedral >::dim_i_t::value >(
                    m_first_pos[0], m_block_id[0]);

                // initialize color dim
                it_domain.template initialize< grid_traits_from_id< enumtype::icosahedral >::dim_c_t::value >(0);
                it_domain.template initialize< grid_traits_from_id< enumtype::icosahedral >::dim_j_t::value >(
                    m_first_pos[1], m_block_id[1]);
                it_domain.template initialize< grid_traits_from_id< enumtype::icosahedral >::dim_k_t::value >(
                    m_grid.template value_at< typename iteration_policy_t::from >());

                int addon = 0;
                // the iterate domain over vertexes has one more grid point
                // TODO specify the loop bounds from the grid_tolopogy to avoid this hack here
                if (location_type_t::value == grid_topology_t::vertexes::value) {
                    addon++;
                }

                typedef color_execution_functor< RunFunctorArguments, iterate_domain_t, grid_t > PP;
                for (uint_t i = m_first_pos[0]; i <= m_first_pos[0] + m_loop_size[0]; ++i) {
                    boost::mpl::for_each< boost::mpl::range_c< uint_t, 0, n_colors_t::value > >(
                        color_execution_functor< RunFunctorArguments, iterate_domain_t, grid_t >(
                            it_domain, m_grid, m_first_pos, m_loop_size, addon));

                    it_domain.template increment< grid_traits_from_id< enumtype::icosahedral >::dim_c_t::value,
                        static_int< -((int_t)n_colors_t::value) > >();

                    it_domain.template increment< grid_traits_from_id< enumtype::icosahedral >::dim_i_t::value,
                        static_int< 1 > >();
                }
                it_domain.template increment< grid_traits_from_id< enumtype::icosahedral >::dim_i_t::value >(
                    -(m_loop_size[0] + 1));
            }

          private:
            const local_domain_t &m_local_domain;
            const grid_t &m_grid;
            const gridtools::array< const uint_t, 2 > m_first_pos;
            const gridtools::array< const uint_t, 2 > m_loop_size;
            const gridtools::array< const uint_t, 2 > m_block_id;
        };

    } // namespace icgrid
} // namespace gridtools
