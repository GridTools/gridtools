#pragma once
#include <boost/utility/enable_if.hpp>
#include "../run_esf_functor.hpp"
#include "../block_size.hpp"
#include "../iterate_domain_evaluator.hpp"

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
        GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), "Internal Error: wrong type");
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
        GRIDTOOLS_STATIC_ASSERT((physical_domain_block_size_t::i_size_t::value ==
                                 processing_elements_block_size_t::i_size_t::value), "Internal Error: wrong type");

        typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;

        using super::m_iterate_domain;

        GT_FUNCTION
        explicit run_esf_functor_cuda(iterate_domain_t& iterate_domain) : super(iterate_domain) {}

        /*
         * @brief main functor implemenation that executes (for CUDA) the user functor of an ESF
         * @tparam IntervalType interval where the functor gets executed
         * @tparam EsfArgument esf arguments type that contains the arguments needed to execute this ESF.
         */
        template<typename IntervalType, typename EsfArguments>
        __device__
        void do_impl() const
        {
            GRIDTOOLS_STATIC_ASSERT((is_esf_arguments<EsfArguments>::value), "Internal Error: wrong type");

            //instantiate the iterate domain evaluator, that will map the calls to arguments to their actual
            // position in the iterate domain
            typedef typename get_iterate_domain_evaluator<iterate_domain_t, typename EsfArguments::esf_args_map_t>::type
                    iterate_domain_evaluator_t;

            iterate_domain_evaluator_t iterate_domain_evaluator(m_iterate_domain);

            typedef typename EsfArguments::functor_t functor_t;

            //a grid point at the core of the block can be out of range (for last blocks) if domain of computations
            // is not a multiple of the block size
            if(m_iterate_domain.is_thread_in_domain())
            {
                //call the user functor at the core of the block
                functor_t::Do(iterate_domain_evaluator, IntervalType());
            }

            this->template execute_extra_work<
                multiple_grid_points_per_warp_t,
                IntervalType,
                EsfArguments,
                iterate_domain_evaluator_t
            > (iterate_domain_evaluator);

            __syncthreads();

        }

    private:

        /*
         * @brief executes the extra grid points associated with each CUDA thread.
         * This extra grid points can be located at the IMinus or IPlus halos or be one of
         * the last J positions in the core of the block
         * @tparam MultipleGridPointsPerWarp boolean template parameter that determines whether a CUDA
         *         thread has to execute more than one grid point (in case of false, the implementation
         *         of this function is empty)
         * @tparam IntervalType type of the interval
         * @tparam EsfArgument esf arguments type that contains the arguments needed to execute this ESF.
         * @tparam IterateDomainEvaluator an iterate domain evaluator that wraps an iterate domain
         */
        template<
            typename MultipleGridPointsPerWarp,
            typename IntervalType,
            typename EsfArguments,
            typename IterateDomainEvaluator
        >
        __device__
        void execute_extra_work(const IterateDomainEvaluator& iterate_domain_evaluator,
                typename boost::disable_if<MultipleGridPointsPerWarp, int >::type=0) const
        {}

        /*
         * @brief executes the extra grid points associated with each CUDA thread.
         * This extra grid points can be located at the IMinus or IPlus halos or be one of
         * the last J positions in the core of the block
         * @tparam MultipleGridPointsPerWarp boolean template parameter that determines whether a CUDA
         *         thread has to execute more than one grid point (in case of false, the implementation
         *         of this function is empty)
         * @tparam IntervalType type of the interval
         * @tparam EsfArgument esf arguments type that contains the arguments needed to execute this ESF.
         * @tparam IterateDomainEvaluator an iterate domain evaluator that wraps an iterate domain
         */
        template<
            typename MultipleGridPointsPerWarp,
            typename IntervalType,
            typename EsfArguments,
            typename IterateDomainEvaluator
        >
        __device__
        void execute_extra_work(const IterateDomainEvaluator& iterate_domain_evaluator,
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
                        (m_iterate_domain).increment<1>(range_t::jminus::value);
                        functor_t::Do(iterate_domain_evaluator, IntervalType());
                        (m_iterate_domain).increment<1>(-range_t::jminus::value);
                    }
                }
                //JPlus halo
                else if(range_t::jplus::value != 0 && (threadIdx.y < -range_t::jminus::value + range_t::jplus::value))
                {
                    if(m_iterate_domain.is_thread_in_domain_x())
                    {
                        const int joffset = range_t::jminus::value + m_iterate_domain.block_size_j();

                        (m_iterate_domain).increment<1>(joffset);
                        functor_t::Do(iterate_domain_evaluator, IntervalType());
                        (m_iterate_domain).increment<1>(-joffset);
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
                        (m_iterate_domain).increment < 0 > (ioffset);
                        (m_iterate_domain).increment < 1 > (joffset);
                        functor_t::Do(iterate_domain_evaluator, IntervalType());
                        (m_iterate_domain).increment < 0 > (-ioffset);
                        (m_iterate_domain).increment < 1 > (-joffset);
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
                        (m_iterate_domain).increment<0>(ioffset);
                        (m_iterate_domain).increment<1>(joffset);
                        functor_t::Do(iterate_domain_evaluator, IntervalType());
                        (m_iterate_domain).increment<0>(-ioffset);
                        (m_iterate_domain).increment<1>(-joffset);
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
                        (m_iterate_domain).increment<1>(joffset);
                        functor_t::Do(iterate_domain_evaluator, IntervalType());
                        (m_iterate_domain).increment<1>(-joffset);
                    }
                }
            }
        }

    };
}
