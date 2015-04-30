#pragma once

#include "../execution_policy.h"
#include "../heap_allocated_temps.h"
#include "../run_kernel.h"
#include "backend_traits.h"

#include "../iteration_policy.h"
#include "../../common/gridtools_runtime.h"
#include "../loop_hierarchy.h"

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


            typedef _impl::run_functor < run_functor_host < Arguments > > base_type;
            explicit run_functor_host(typename Arguments::local_domain_list_t& domain_list,  typename Arguments::coords_t const& coords)
                : base_type(domain_list, coords)
                {}

            explicit run_functor_host(typename Arguments::local_domain_list_t& domain_list
                                      , typename Arguments::coords_t const& coords
                                      , uint_t i
                                      , uint_t j
                                      , uint_t bi
                                      , uint_t bj
                                      , uint_t blki
                                      , uint_t blkj)
                : base_type(domain_list, coords, i, j, bi, bj, blki, blkj)
                {}

        };
    }

    namespace _impl {
        template<typename Arguments>
        struct run_functor_backend_id<_impl_host::run_functor_host<Arguments> > :
            boost::mpl::integral_c<enumtype::backend, enumtype::Host> {};
    } // namespace _impl{

    /** @brief Partial specialization: naive and block implementation for the host backend */
    template <typename Arguments >
    struct execute_kernel_functor < _impl_host::run_functor_host< Arguments > >
    {
        typedef _impl_host::run_functor_host< Arguments > backend_t;

        /**
           @brief functor implementing the kernel executed in the innermost loop

           This functor contains the portion of the code executed in the innermost loop. In this case it
           is the loop over the third dimension (k), but the generality of the loop hierarchy implementation
           allows to easily generalize this.
        */
        template<typename LoopIntervals, typename RunOnInterval, typename IterateDomain, typename RunKernelType, typename  IterationPolicy>
        struct innermost_functor{

        private:

            IterateDomain & m_it_domain;
            RunKernelType const* m_functor;

        public:

            IterateDomain const& it_domain() const { return m_it_domain; }
            RunKernelType const* functor() const { return m_functor; }

            innermost_functor(IterateDomain & it_domain, RunKernelType const* f):
                m_it_domain(it_domain),
                m_functor(f){}

            void operator() () const {
                m_it_domain.template set_k_start( m_functor->m_coords.template value_at< typename IterationPolicy::from >() );
                gridtools::for_each< LoopIntervals >
                    ( RunOnInterval (m_it_domain,m_functor->m_coords) );
            }
        };

        /**
           @brief core of the kernel execution
           \tparam Traits traits class defined in \ref gridtools::_impl::run_functor_traits
        */
        template< typename EsfArguments >
        static void execute_kernel( typename EsfArguments::local_domain_t& local_domain, const backend_t * f )
        {
            BOOST_STATIC_ASSERT((is_esf_arguments<EsfArguments>::value));

            typedef typename Arguments::coords_t coords_type;
            typedef typename Arguments::loop_intervals_t loop_intervals_t;
            typedef typename Arguments::execution_type_t execution_type_t;
            typedef typename EsfArguments::range_t range_t;
            typedef typename EsfArguments::functor_t functor_type;
            typedef typename EsfArguments::iterate_domain_t iterate_domain_t;
            typedef typename EsfArguments::first_hit_t first_hit_t;

            typedef backend_traits_from_id<enumtype::Host> backend_traits_t;
#ifndef NDEBUG
            std::cout << "Functor " <<  functor_type() << "\n";
            std::cout << "I loop " << (int_t)f->m_starti <<"+"<< range_t::iminus::value << " -> "
                      << f->m_starti <<"+"<< f->m_BI <<"+"<< range_t::iplus::value << "\n";
            std::cout << "J loop " << (int_t)f->m_startj <<"+"<< range_t::jminus::value << " -> "
                      << (int_t)f->m_startj <<"+"<< f->m_BJ <<"+"<< range_t::jplus::value << "\n";
            std::cout <<  " ******************** " << first_hit_t() << "\n";
            std::cout << " ******************** " << f->m_coords.template value_at<first_hit_t>() << "\n";
            std::cout<<"iminus::value: "<<range_t::iminus::value<<std::endl;
#endif

            array<void* RESTRICT,iterate_domain_t::N_DATA_POINTERS> data_pointer;
            strides_cached<iterate_domain_t::N_STORAGES-1, typename Traits::local_domain_t::esf_args> strides;

            iterate_domain_t it_domain(local_domain);
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
                (int_t) (func_->m_start[0]),
                (int_t) (func_->m_start[0] + func_->m_block[0]),
                (int_t) (func_->m_start[1]),
                (int_t) (func_->m_start[1] + func_->m_block[1])
            );

                //reset the index
            it_domain.set_index(0);
            ij_loop.initialize(it_domain, func_->m_block_id);

            //define the kernel functor
            typedef innermost_functor<
                loop_intervals_t,
                _impl::run_f_on_interval<
                    execution_type_t,
                    EsfArguments, 
                    Arguments
                >,
                iterate_domain_t,
                backend_t,
                iteration_policy
            > innermost_functor_t;

            //instantiate the kernel functor
            innermost_functor_t f(it_domain,func_);

            //run the nested ij loop
            ij_loop.apply(it_domain, f);

        }

    };


    /**
       @brief given the backend \ref gridtools::_impl_host::run_functor_host returns the backend ID gridtools::enumtype::Host
       wasted code because of the lack of constexpr
    */

    //// Check if this is needed
    template <typename Arguments >
    struct backend_type< _impl_host::run_functor_host< Arguments > >
    {
        static const enumtype::backend s_backend=enumtype::Host;
    };

} // namespace gridtools
