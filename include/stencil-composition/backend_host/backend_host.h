#pragma once

#include "../execution_policy.h"
#include "../heap_allocated_temps.h"
#include "../run_kernel.h"
#include "backend_traits.h"

#include "../iteration_policy.h"
#include "../../common/gridtools_runtime.h"

/**
   @file
   @brief Implements the stencil operations for the host backend
*/

namespace gridtools {

//    namespace _impl_host {
//
//        /** @brief Host backend
//            Derived class of the CRTP pattern defined in \ref gridtools::_impl::run_functor */
//        template < typename Arguments >
//        struct run_functor_host : public _impl::run_functor < run_functor_host< Arguments > >
//        {
//
//            typedef _impl::run_functor < run_functor_host < Arguments > > base_type;
//            explicit run_functor_host(typename Arguments::local_domain_list_t& domain_list,  typename Arguments::coords_t const& coords)
//                : base_type(domain_list, coords)
//            {}
//
//            explicit run_functor_host(typename Arguments::local_domain_list_t& domain_list,  typename Arguments::coords_t const& coords, uint_t i, uint_t j, uint_t bi, uint_t bj, uint_t blki, uint_t blkj)
//                : base_type(domain_list, coords, i, j, bi, bj, blki, blkj)
//            {}
//
//        };
//    }
//
//    namespace _impl {
//        template<typename Arguments>
//        struct run_functor_backend_id<_impl_host::run_functor_host<Arguments> > :
//            boost::mpl::integral_c<enumtype::backend, enumtype::Host> {};
//    } // namespace _impl{
//
//    /** @brief Partial specialization: naive and block implementation for the host backend */
//    template <typename Arguments >
//    struct execute_kernel_functor < _impl_host::run_functor_host< Arguments > >
//    {
//        typedef _impl_host::run_functor_host< Arguments > backend_t;
//
//        /**
//           @brief core of the kernel execution
//           \tparam Traits traits class defined in \ref gridtools::_impl::run_functor_traits
//        */
//        template< typename EsfArguments >
//        static void execute_kernel( typename EsfArguments::local_domain_t& local_domain, const backend_t * f )
//        {
//            BOOST_STATIC_ASSERT((is_esf_arguments<EsfArguments>::value));
//
//            typedef typename Arguments::coords_t coords_type;
//            typedef typename Arguments::loop_intervals_t loop_intervals_t;
//            typedef typename Arguments::execution_type_t execution_type_t;
//            typedef typename EsfArguments::range_t range_t;
//            typedef typename EsfArguments::functor_t functor_type;
//            typedef typename EsfArguments::iterate_domain_t iterate_domain_t;
//            typedef typename EsfArguments::first_hit_t first_hit_t;
//
//            typedef backend_traits_from_id<enumtype::Host> backend_traits_t;
//#ifndef NDEBUG
//            std::cout << "Functor " <<  functor_type() << "\n";
//            std::cout << "I loop " << (int_t)f->m_starti <<"+"<< range_t::iminus::value << " -> "
//                      << f->m_starti <<"+"<< f->m_BI <<"+"<< range_t::iplus::value << "\n";
//            std::cout << "J loop " << (int_t)f->m_startj <<"+"<< range_t::jminus::value << " -> "
//                      << (int_t)f->m_startj <<"+"<< f->m_BJ <<"+"<< range_t::jplus::value << "\n";
//            std::cout <<  " ******************** " << first_hit_t() << "\n";
//            std::cout << " ******************** " << f->m_coords.template value_at<first_hit_t>() << "\n";
//            std::cout<<"iminus::value: "<<range_t::iminus::value<<std::endl;
//#endif
//
//            void* data_pointer[iterate_domain_t::N_DATA_POINTERS];
//            iterate_domain_t it_domain(local_domain);
//
//            it_domain.template assign_storage_pointers<backend_traits_t >(data_pointer);
//
//            for (int_t i = (int_t)f->m_starti + range_t::iminus::value;
//                 i <= (int_t)f->m_starti + (int_t)f->m_BI + range_t::iplus::value;
//                 ++i)
//            {
//                // for_each<local_domain.local_args>(increment<0>);
//                for (int_t j = (int_t)f->m_startj + range_t::jminus::value;
//                        j <= (int_t)f->m_startj + (int_t)f->m_BJ + range_t::jplus::value; ++j)
//                {
//                    // for_each<local_domain.local_args>(increment<1>());
//                    //#ifndef NDEBUG
//                    //std::cout << "Move to : " << i << ", " << j << std::endl;
//                    //#endif
//                    //reset the index
//                    it_domain.set_index(0);
//                    it_domain.template assign_ij<0>(i, f->blk_idx_i);
//                    it_domain.template assign_ij<1>(j, f->blk_idx_j);
//                    /** setting an iterator to the address of the current i,j entry to be accessed */
//                    typedef typename boost::mpl::front<loop_intervals_t>::type interval;
//                    typedef typename index_to_level<typename interval::first>::type from;
//                    typedef typename index_to_level<typename interval::second>::type to;
//                    typedef _impl::iteration_policy<from, to, execution_type_t::type::iteration> iteration_policy;
//                    assert(i>=0);
//                    assert(j>=0);
//
//                    //printf("setting the start to: %d \n",f->m_coords.template value_at< typename iteration_policy::from >() );
//                    //setting the initial k level (for backward/parallel iterations it is not 0)
//                    it_domain.set_k_start( f->m_coords.template value_at< typename iteration_policy::from >() );
//
//                    //local structs can be passed as template arguments in C++11 (would improve readability)
//
//                    /** run the iteration on the k dimension */
//                    gridtools::for_each< loop_intervals_t > (
//                        _impl::run_f_on_interval<
//                            execution_type_t, EsfArguments, Arguments
//                         >(it_domain,f->m_coords) );
//                }
//            }
//        }
//
//    };


//    /** @brief Partial specialization: naive and block implementation for the host backend */
//    template <typename RunFunctorArguments >
//    struct execute_kernel_functor_host
//    {
//        BOOST_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value));
//        typedef typename RunFunctorArguments::local_domain_list_t local_domain_list_t;
//        typedef typename RunFunctorArguments::coords_t coords_t;
//
//        BOOST_STATIC_ASSERT((boost::mpl::size<local_domain_list_t>::value==1));
//        typedef typename boost::mpl::back<local_domain_list_t>::type local_domain_t;
//        BOOST_STATIC_ASSERT((is_local_domain<local_domain_t>::value));
//
//        /**
//           @brief core of the kernel execution
//           \tparam Traits traits class defined in \ref gridtools::_impl::run_functor_traits
//        */
//        explicit execute_kernel_functor_host(const local_domain_list_t& local_domain_list, const coords_t& coords,
//                const uint_t starti, const uint_t startj, const uint_t block_size_i, const uint_t block_size_j,
//                const uint_t block_idx_i, const uint_t block_idx_j)
//        : m_local_domain(boost::fusion::at<0>(local_domain_list) )
//        , m_coords(coords)
//        , m_starti(starti)
//        , m_startj(startj)
//        , m_block_size_i(block_size_i)
//        , m_block_size_j(block_size_j)
//        , m_block_idx_i(block_idx_i)
//        , m_block_idx_j(block_idx_j)
//        {
//
//
//        }
//
//        // Naive strategy
//        explicit  execute_kernel_functor_host(const local_domain_list_t& local_domain_list, const coords_t& coords)
//            : m_local_domain(boost::fusion::at<0>(local_domain_list) )
//        , m_coords(coords)
//        , m_starti(coords.i_low_bound())
//        , m_startj(coords.j_low_bound())
//        , m_block_size_i(coords.i_high_bound()-coords.i_low_bound())
//        , m_block_size_i(coords.j_high_bound()-coords.j_low_bound())
//        , m_block_idx_i(0)
//        , m_block_idx_j(0)
//
//        {}
//
//        void operator()()
//        {
//            typedef typename RunFunctorArguments::coords_t coords_type;
//            typedef typename RunFunctorArguments::loop_intervals_t loop_intervals_t;
//            typedef typename RunFunctorArguments::execution_type_t execution_type_t;
//
//            // in the host backend there should be only one esf per mss
//            BOOST_STATIC_ASSERT((boost::mpl::size<typename RunFunctorArguments::range_sizes_t>::value==1));
//            typedef typename boost::mpl::back<typename RunFunctorArguments::range_sizes_t>::type range_t;
//            BOOST_STATIC_ASSERT((is_range<range_t>::value));
//
//            typedef typename local_domain_t::iterate_domain_t iterate_domain_t;
//
//            typedef backend_traits_from_id<enumtype::Host> backend_traits_t;
//#ifndef NDEBUG
//            std::cout << "I loop " << m_starti <<"+"<< range_t::iminus::value << " -> "
//                      << m_starti <<"+"<< m_block_size_i <<"+"<< range_t::iplus::value << "\n";
//            std::cout << "J loop " << m_startj <<"+"<< range_t::jminus::value << " -> "
//                      << m_startj <<"+"<< m_block_size_j <<"+"<< range_t::jplus::value << "\n";
//            std::cout<<"iminus::value: "<<range_t::iminus::value<<std::endl;
//#endif
//
//            void* data_pointer[iterate_domain_t::N_DATA_POINTERS];
//            iterate_domain_t it_domain(m_local_domain);
//
//            it_domain.template assign_storage_pointers<backend_traits_t >(data_pointer);
//
//            for (int_t i = m_starti + range_t::iminus::value;
//                 i <= m_starti + m_block_size_i + range_t::iplus::value; ++i)
//            {
//                for (int_t j = m_startj + range_t::jminus::value;
//                    j <= m_startj + m_block_size_j + range_t::jplus::value; ++j)
//                {
//                    //reset the index
//                    it_domain.set_index(0);
//                    it_domain.template assign_ij<0>(i, m_block_idx_i);
//                    it_domain.template assign_ij<1>(j, m_block_idx_j);
//                    /** setting an iterator to the address of the current i,j entry to be accessed */
//                    typedef typename boost::mpl::front<loop_intervals_t>::type interval;
//                    typedef typename index_to_level<typename interval::first>::type from;
//                    typedef typename index_to_level<typename interval::second>::type to;
//                    typedef _impl::iteration_policy<from, to, execution_type_t::type::iteration> iteration_policy;
//                    assert(i>=0);
//                    assert(j>=0);
//
//                    //setting the initial k level (for backward/parallel iterations it is not 0)
//                    it_domain.set_k_start( m_coords.template value_at< typename iteration_policy::from >() );
//
//                //local structs can be passed as template arguments in C++11 (would improve readability)
//
////                /** run the iteration on the k dimension */
////                gridtools::for_each< loop_intervals_t > (
////                    _impl::run_f_on_interval<
////                            execution_type_t, EsfArguments, RunFunctorArguments
////                         >(it_domain,f->m_coords) );
////                }
//                }
//            }
//        }
//    private:
//        const local_domain_t& m_local_domain;
//        const coords_t& m_coords;
//        const uint_t m_starti;
//        const uint_t m_startj;
//        const uint_t m_block_size_i;
//        const uint_t m_block_size_j;
//        const uint_t m_block_idx_i;
//        const uint_t m_block_idx_j;
//
//    };


//    /**
//       @brief given the backend \ref gridtools::_impl_host::run_functor_host returns the backend ID gridtools::enumtype::Host
//       wasted code because of the lack of constexpr
//    */
//
//    //// Check if this is needed
//    template <typename Arguments >
//    struct backend_type< _impl_host::run_functor_host< Arguments > >
//    {
//        static const enumtype::backend s_backend=enumtype::Host;
//    };

    // } //namespace _impl

} // namespace gridtools
