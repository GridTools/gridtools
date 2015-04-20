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

    namespace _impl_host {

        /** @brief Host backend
            Derived class of the CRTP pattern defined in \ref gridtools::_impl::run_functor */
        template < typename Arguments >
        struct run_functor_host : public _impl::run_functor < run_functor_host< Arguments > >
        {

            typedef _impl::run_functor < run_functor_host < Arguments > > base_type;
            explicit run_functor_host(typename Arguments::local_domain_list_t& domain_list,  typename Arguments::coords_t const& coords)
                : base_type(domain_list, coords)
            {}

            explicit run_functor_host(typename Arguments::local_domain_list_t& domain_list,  typename Arguments::coords_t const& coords, uint_t i, uint_t j, uint_t bi, uint_t bj, uint_t blki, uint_t blkj)
                : base_type(domain_list, coords, i, j, bi, bj, blki, blkj)
            {}

        };
    }

    // namespace _impl{

    /** @brief Partial specialization: naive and block implementation for the host backend */
    template <typename Arguments >
    struct execute_kernel_functor < _impl_host::run_functor_host< Arguments > >
    {
        typedef _impl_host::run_functor_host< Arguments > backend_t;

        template <typename FunctorType, typename IntervalMapType, typename IterateDomainType, typename CoordsType>
        struct extra_arguments{
            typedef FunctorType functor_t;
            typedef IntervalMapType interval_map_t;
            typedef IterateDomainType local_domain_t;
            typedef CoordsType coords_t;
        };

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
        template< typename Traits >
        static void execute_kernel( typename Traits::local_domain_t& local_domain, const backend_t * f )
        {
            typedef typename Arguments::coords_t coords_type;
            typedef typename Arguments::loop_intervals_t loop_intervals_t;
            typedef typename Traits::range_t range_t;
            typedef typename Traits::functor_t functor_type;
            typedef typename Traits::local_domain_t  local_domain_t;
            typedef typename Traits::interval_map_t interval_map_type;
            typedef typename Traits::iterate_domain_t iterate_domain_type;
            typedef typename Arguments::execution_type_t execution_type_t;

            typedef typename boost::mpl::eval_if_c<has_xrange<functor_type>::type::value, get_xrange< functor_type >, boost::mpl::identity<range<0,0,0> > >::type new_range_t;
            typedef typename sum_range<new_range_t, range_t>::type xrange_t;
            typedef typename boost::mpl::eval_if_c<has_xrange_subdomain<functor_type>::type::value, get_xrange_subdomain< functor_type >, boost::mpl::identity<range<0,0,0> > >::type xrange_subdomain_t;

            int_t boundary=f->m_coords.partitioner()/*.communicator()*/.boundary();
            int_t jminus=(int_t)  xrange_subdomain_t::jminus::value + ((boundary)>7? xrange_t::jminus::value : 0) ;//j-low
            int_t iminus=(int_t) (xrange_subdomain_t::iminus::value + ((boundary%8)>3? xrange_t::iminus::value : 0) );//i-low
            int_t jplus=(int_t)  (xrange_subdomain_t::jplus::value + ((boundary%4)>1? xrange_t::jplus::value : 0)) ;//j-high
            int_t iplus=(int_t) xrange_subdomain_t::iplus::value + ((boundary%2)>0? xrange_t::iplus::value : 0) ;//i-high

            typedef backend_traits_from_id<enumtype::Host> backend_traits_t;
#ifndef NDEBUG
            std::cout << "Functor " <<  functor_type() << "\n";
            std::cout << "I loop " << (int_t)f->m_start[0] <<"+"<< iminus << " -> "
                      << f->m_start[0] <<"+"<< f->m_block[0] <<"+"<< iplus << "\n";
            std::cout << "J loop " << (int_t)f->m_start[1] <<"+"<< jminus << " -> "
                      << (int_t)f->m_start[1] <<"+"<< f->m_block[1] <<"+"<< jplus << "\n";
            std::cout <<  " ******************** " << typename Traits::first_hit_t() << "\n";
            std::cout << " ******************** " << f->m_coords.template value_at<typename Traits::first_hit_t>() << "\n";
            std::cout<<"iminus::value: "<<iminus<<std::endl;
#endif

            array<void* __restrict__,Traits::iterate_domain_t::N_DATA_POINTERS> data_pointer;
            storage_cached<Traits::iterate_domain_t::N_STORAGES-1, typename Traits::local_domain_t::esf_args> strides;

             iterate_domain_type it_domain(local_domain);
             it_domain.template assign_storage_pointers<backend_traits_t >(&data_pointer);

             it_domain.template assign_stride_pointers <backend_traits_from_id<enumtype::Host> >(&strides);

             typedef typename boost::mpl::front<loop_intervals_t>::type interval;
             typedef typename index_to_level<typename interval::first>::type from;
             typedef typename index_to_level<typename interval::second>::type to;
             typedef _impl::iteration_policy<from, to, execution_type_t::type::iteration> iteration_policy;

#ifdef NEW_IMPLEMENTATION

             typedef array<int_t, Traits::iterate_domain_t::N_STORAGES> array_t;
                    loop_hierarchy<array_t, loop_item<0, enumtype::forward>, loop_item<1, enumtype::forward> > ij_loop(
                        f->m_start[0] + iminus,
                        f->m_start[0] + f->m_block[0] + iplus,
                        f->m_start[1] + jminus,
                        f->m_start[1] + f->m_block[1] + jplus
                        );

                    //reset the index
                    it_domain.set_index(0);
                    ij_loop.initialize(it_domain, f->m_block_id);

                    typedef innermost_functor<loop_intervals_t
                                              , _impl::run_f_on_interval
                                              <
                                                  execution_type_t,
                                                  extra_arguments<functor_type, interval_map_type, iterate_domain_type, coords_type>
                                                  >
                                              , typename Traits::iterate_domain_t
                                              , backend_t
                                              , iteration_policy
                                              > innermost_functor_t;

                    ij_loop.apply(it_domain, innermost_functor_t(it_domain,f));

#else
             it_domain.set_index(0);
             it_domain.template initialize<0>((int_t)f->m_start[0] + iminus, f->m_block_id[0]);
             it_domain.template initialize<1>((int_t)f->m_start[1] + jminus, f->m_block_id[1]);
             array<int_t, Traits::iterate_domain_t::N_STORAGES> restore_index_j(0);
             array<int_t, Traits::iterate_domain_t::N_STORAGES> restore_index_k(0);

             //initialization
             it_domain.get_index(restore_index_j);
             it_domain.get_index(restore_index_k);



                    for (int_t i = (int_t)f->m_start[0] + iminus;
                         i <= (int_t)f->m_start[0] + (int_t)f->m_block[0] + iplus;
                         ++i)
                    {
                        // for_each<local_domain.local_args>(increment<0>);
                        for (int_t j = (int_t)f->m_start[1] + jminus;
                             j <= (int_t)f->m_start[1] + (int_t)f->m_block[1] + jplus;
                             ++j)
                        {
                            // it_domain.template assign_ij<2>( f->m_coords.template value_at< typename iteration_policy::from >());

                            assert(i>=0);
                            assert(j>=0);


                            /** setting an iterator to the address of the current i,j entry to be accessed */
                            it_domain.set_k_start( f->m_coords.template value_at< typename iteration_policy::from >() );

                            //local structs can be passed as template arguments in C++11 (would improve readability)
                            // struct extra_arguments{
                            //     typedef functor_type functor_t;
                            //     typedef interval_map_type interval_map_t;
                            //     typedef iterate_domain_type local_domain_t;
                            //     typedef coords_type coords_t;};

                            /** run the iteration on the k dimension */
                            gridtools::for_each< loop_intervals_t >
                                (_impl::run_f_on_interval
                                 <
                                 execution_type_t,
                                 extra_arguments<functor_type, interval_map_type, iterate_domain_type, coords_type>
                                 >
                                 (it_domain,f->m_coords)
                                 );
                            //restore the k index
                            it_domain.set_index(restore_index_k);
                            it_domain.template increment<1, enumtype::forward>();
                            // it_domain.template increment<1, enumtype::forward>(1, f->blk_idx_j);
                            it_domain.get_index(restore_index_k);//redundant in the last iteration

                        }
                        //restore the j index
                        it_domain.set_index(restore_index_j);
                        //increment it
                        it_domain.template increment<0, enumtype::forward>();
                        // it_domain.template increment<0, enumtype::forward>(1, f->blk_idx_i);
                        //save the new value
                        it_domain.get_index(restore_index_j);
                        it_domain.get_index(restore_index_k);
                }
#endif
        }

    };


} // namespace gridtools
