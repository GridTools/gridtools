#pragma once
#include "backend.h"

namespace gridtools {
    namespace _impl {

        template<typename RunFunctorImpl>
        struct run_functor_backend_id;

//        /**
//           \brief "base" struct for all the backend
//           This class implements static polimorphism by means of the CRTP pattern. It contains all what is common for all the backends.
//        */
//        template < typename RunFunctorImpl >
//	    struct run_functor {
//            typedef typename run_functor_impl_arguments<RunFunctorImpl>::type run_functor_arguments_t;
//            typedef typename run_functor_arguments_t::local_domain_list_t local_domain_list_t;
//            typedef typename run_functor_arguments_t::coords_t coords_t;
//            typedef typename run_functor_arguments_t::esf_args_map_sequence_t esf_args_map_sequence_t;
//            typedef typename run_functor_arguments_t::functor_list_t run_functor_list_t;
//
//            local_domain_list_t & m_local_domain_list;
//            coords_t const & m_coords;
//            const uint_t m_starti, m_startj, m_BI, m_BJ, blk_idx_i, blk_idx_j;
//
//            // Block strategy
//            explicit run_functor(local_domain_list_t& dom_list,
//                                 coords_t const& coords,
//                                 uint_t i, uint_t j,
//                                 uint_t bi, uint_t bj,
//                                 uint_t blk_idx_i, uint_t blk_idx_j)
//                : m_local_domain_list(dom_list)
//                , m_coords(coords)
//                , m_starti(i)
//                , m_startj(j)
//                , m_BI(bi)
//                , m_BJ(bj)
//                , blk_idx_i(blk_idx_i)
//                , blk_idx_j(blk_idx_j)
//            {}
//
//            // Naive strategy
//            explicit run_functor(local_domain_list_t& dom_list,
//                                 coords_t const& coords)
//                : m_local_domain_list(dom_list)
//                , m_coords(coords)
//                , m_starti(coords.i_low_bound())
//                , m_startj(coords.j_low_bound())
//                , m_BI(coords.i_high_bound()-coords.i_low_bound())
//                , m_BJ(coords.j_high_bound()-coords.j_low_bound())
//                , blk_idx_i(0)
//                , blk_idx_j(0)
//            {}
//
//            /**
//             * \brief given the index of a functor in the functors
//             * list, it calls a kernel on the GPU executing the
//             * operations defined on that functor.
//             */
//            template <typename Index>
//            void operator()(Index const& ) const {
//
//                typedef esf_arguments<
//                    typename run_functor_backend_id<RunFunctorImpl>::type,
//                    run_functor_arguments_t,
//                    Index
//                > esf_arguments_t;
//
//                typedef typename esf_arguments_t::local_domain_index_t local_domain_index_t;
//                typedef typename esf_arguments_t::local_domain_t local_domain_t;
//
//                local_domain_t& local_domain = boost::fusion::at<local_domain_index_t>(m_local_domain_list);
//
//                typedef execute_kernel_functor< RunFunctorImpl > exec_functor_t;
//                typedef typename boost::mpl::at<esf_args_map_sequence_t, Index>::type esf_args_map_t;
//
//                //check that the number of placeholders passed to the elementary stencil function
//                //(constructed during the computation) is the same as the number of arguments referenced
//                //in the functor definition (in the high level interface). This means that we cannot
//                // (although in theory we could) pass placeholders to the computation which are not
//                //also referenced in the functor.
//                GRIDTOOLS_STATIC_ASSERT( (boost::mpl::size<esf_args_map_t>::value==
//                    boost::mpl::size<typename boost::mpl::at<run_functor_list_t, Index>::type::arg_list>::value ),
//		            "GRIDTOOLS ERROR:\n\
//		            check that the number of placeholders passed to the elementary stencil function\n \
//		            (constructed during the computation) is the same as the number of arguments referenced\n\
//		            in the functor definition (in the high level interface). This means that we cannot\n\
//		            (although in theory we could) pass placeholders to the computation which are not\n\
//		            also referenced in the functor.");
//
//                exec_functor_t::template execute_kernel<
//                    esf_arguments_t
//                >(local_domain, static_cast<const RunFunctorImpl*>(this));
//            }
//        };
    } // namespace _impl
} // namespace gridtools
