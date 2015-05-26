#pragma once
#include <boost/utility/enable_if.hpp>
#include "../run_esf_functor.h"
#include "../block_size.h"
#include "../iterate_domain_evaluator.h"

namespace gridtools {
    /*
     * @brief main functor that executes (for CUDA) the user functor of an ESF
     * @tparam RunFunctorArguments run functor arguments
     * @tparam Interval interval where the functor gets executed
     */
    template < typename RunFunctorArguments, typename Interval>
    struct run_esf_functor_cuda :
            public run_esf_functor<run_esf_functor_cuda<RunFunctorArguments, Interval> > //CRTP
    {
        BOOST_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value));
        //TODOCOSUNA This type here is not an interval, is a pair<int_, int_ >
        //BOOST_STATIC_ASSERT((is_interval<Interval>::value));

        typedef run_esf_functor<run_esf_functor_cuda<RunFunctorArguments, Interval> > super;
        typedef typename RunFunctorArguments::physical_domain_block_size_t physical_domain_block_size_t;
        typedef typename RunFunctorArguments::processing_elements_block_size_t processing_elements_block_size_t;

        //metavalue that determines if a warp is processing more grid points that the default assigned
        // at the core of the block
        typedef typename boost::mpl::not_<
            typename boost::is_same<physical_domain_block_size_t, processing_elements_block_size_t>::type
        >::type multiple_grid_points_per_warp_t;

        //nevertheless, even if each thread computes more than a grid point, the i size of the physical block
        //size and the cuda block size have to be the same
        BOOST_STATIC_ASSERT((physical_domain_block_size_t::i_size_t::value ==
                processing_elements_block_size_t::i_size_t::value));

        typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;

        using super::m_iterate_domain;

        GT_FUNCTION
        explicit run_esf_functor_cuda(iterate_domain_t& iterate_domain) : super(iterate_domain) {}

        template<typename IntervalType, typename EsfArguments>
        __device__
        void DoImpl() const
        {
            BOOST_STATIC_ASSERT((is_esf_arguments<EsfArguments>::value));

            //instantiate the iterate domain evaluator, that will map the calls to arguments to their actual

            typedef typename EsfArguments::functor_t functor_t;

            //a grid point at the core of the block can be out of range (for last blocks) if domain of computations
            // is not a multiple of the block size
            if(m_iterate_domain.is_thread_in_domain())
            {
                //call the user functor at the core of the block
                functor_t::Do(m_iterate_domain, IntervalType());
            }

            this->template ExecuteExtraWork<
                multiple_grid_points_per_warp_t,
                IntervalType,
                EsfArguments,
                iterate_domain_t
            > (m_iterate_domain);

            __syncthreads();

        }

    private:

        template<
            typename MultipleGridPointsPerWarp,
            typename IntervalType,
            typename EsfArguments,
            typename IterateDomainEvaluator
        >
        __device__
        void ExecuteExtraWork(const IterateDomainEvaluator& iterate_domain_evaluator,
                typename boost::disable_if<MultipleGridPointsPerWarp, int >::type=0) const
        {}

        template<
            typename MultipleGridPointsPerWarp,
            typename IntervalType,
            typename EsfArguments,
            typename IterateDomainEvaluator
        >
        __device__
        void ExecuteExtraWork(const IterateDomainEvaluator& iterate_domain_evaluator,
                typename boost::enable_if<MultipleGridPointsPerWarp, int >::type=0) const
        {
            typedef typename EsfArguments::functor_t functor_t;
            typedef typename EsfArguments::range_t range_t;

            //if the warps need to compute more grid points than the core of the block
            if(multiple_grid_points_per_warp_t::value) {
                //JMinus  halo
                if(range_t::jminus::value != 0 && (threadIdx.y < -range_t::jminus::value))
                {
                    if(m_iterate_domain.is_thread_in_domain_x())
                    {
                        (m_iterate_domain).advance_ij<1>(range_t::jminus::value);
                        functor_t::Do(iterate_domain_evaluator, IntervalType());
                        (m_iterate_domain).advance_ij<1>(-range_t::jminus::value);
                    }
                }
                //JPlus halo
                else if(range_t::jplus::value != 0 && (threadIdx.y < -range_t::jminus::value + range_t::jplus::value))
                {
                    if(m_iterate_domain.is_thread_in_domain_x())
                    {
                        const int joffset = range_t::jminus::value + m_iterate_domain.block_size_j();

                        (m_iterate_domain).advance_ij<1>(joffset);
                        functor_t::Do(iterate_domain_evaluator, IntervalType());
                        (m_iterate_domain).advance_ij<1>(-joffset);
                    }
                }
                //IMinus halo
                else if(range_t::iminus::value != 0 && (threadIdx.y < -range_t::jminus::value + range_t::jplus::value + 1))
                {
                    const int ioffset = -m_iterate_domain.thread_position_x() -
                        (m_iterate_domain.thread_position_x() % (-range_t::iminus::value))-1;
                    const int joffset = -m_iterate_domain.thread_position_y() +
                        (m_iterate_domain.thread_position_x() / (-range_t::iminus::value) );

                    if(m_iterate_domain.is_thread_in_domain_y(joffset))
                    {
                        (m_iterate_domain).advance_ij < 0 > (ioffset);
                        (m_iterate_domain).advance_ij < 1 > (joffset);
                        functor_t::Do(iterate_domain_evaluator, IntervalType());
                        (m_iterate_domain).advance_ij < 0 > (-ioffset);
                        (m_iterate_domain).advance_ij < 1 > (-joffset);
                    }
                }
                //IPlus halo
                else if(range_t::iplus::value != 0 && (threadIdx.y < -range_t::jminus::value + range_t::jplus::value +
                    (range_t::iminus::value != 0 ? 1 : 0) + 1))
                {
                    const int ioffset = -m_iterate_domain.thread_position_x() +
                        m_iterate_domain.block_size_i() + (threadIdx.x % (range_t::iplus::value));
                    const int joffset = -m_iterate_domain.thread_position_y() +
                        (threadIdx.x / (range_t::iplus::value) );

                    if(m_iterate_domain.is_thread_in_domain_y(joffset))
                    {
                        (m_iterate_domain).advance_ij<0>(ioffset);
                        (m_iterate_domain).advance_ij<1>(joffset);
                        functor_t::Do(iterate_domain_evaluator, IntervalType());
                        (m_iterate_domain).advance_ij<0>(-ioffset);
                        (m_iterate_domain).advance_ij<1>(-joffset);
                    }
                }
                //the remaining warps will compute extra work at the core of the block
                else
                {
                    const int joffset = blockDim.y +
                        range_t::jminus::value - range_t::jplus::value -
                        (range_t::iminus::value != 0 ? 1 : 0) - (range_t::iplus::value != 0 ? 1 : 0);

                    if(m_iterate_domain.is_thread_in_domain(0, joffset))
                    {
                        (m_iterate_domain).advance_ij<1>(joffset);
                        functor_t::Do(iterate_domain_evaluator, IntervalType());
                        (m_iterate_domain).advance_ij<1>(-joffset);
                    }
                }
            }
        }

    };
}
