/*
 * run_esf_functor_host.h
 *
 *  Created on: Apr 27, 2015
 *      Author: cosuna
 */
#pragma once
#include "../run_esf_functor.h"
#include "../block_size.h"
#include "../iterate_domain_evaluator.h"

namespace gridtools {
    template < typename RunFunctorArguments, typename Interval>
    struct run_esf_functor_cuda :
            public run_esf_functor<run_esf_functor_cuda<RunFunctorArguments, Interval> > //CRTP
    {
        BOOST_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value));
        typedef typename RunFunctorArguments::block_size_t block_size_t;
        typedef run_esf_functor<run_esf_functor_cuda<RunFunctorArguments, Interval> > super;
        typedef typename RunFunctorArguments::local_domain_t::iterate_domain_t iterate_domain_t;

        GT_FUNCTION
        explicit run_esf_functor_cuda(iterate_domain_t& iterate_domain) : super(iterate_domain) {}

        template<typename IntervalType, typename EsfArguments>
        __device__
        void DoImpl() const
        {
            printf("PENDOLINO\n");
            BOOST_STATIC_ASSERT((is_esf_arguments<EsfArguments>::value));

            typedef typename get_iterate_domain_evaluator<iterate_domain_t, typename EsfArguments::esf_args_map_t>::type
                    iterate_domain_evaluator_t;

            iterate_domain_evaluator_t iterate_domain_evaluator(this->m_iterate_domain);

            typedef typename EsfArguments::functor_t functor_t;
            typedef typename EsfArguments::range_t range_t;
            functor_t::Do(iterate_domain_evaluator, IntervalType());

            printf("DDPD %d %d %d %d %d \n", blockDim.y, block_size_t::j_size_t::value, block_size_t::i_size_t::value,
                    threadIdx.x, threadIdx.y);

            const int reuse_warps = -range_t::jminus::value + range_t::jplus::value +
                    (range_t::iminus::value != 0 ? 1 : 0) + (range_t::iplus::value !=0 ? 1 : 0);

            const int cuda_block_size_j = (blockDim.y - reuse_warps ) *
                    ((range_t::iminus::value !=0 || range_t::iplus::value != 0) ? 2 : 1) + reuse_warps;

            if(range_t::jminus::value != 0 && (threadIdx.y < -range_t::jminus::value))
            {
                printf("JMINUS %d %d %d %d \n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
                (this->m_iterate_domain).advance_ij<1>(range_t::jminus::value, 0);
                functor_t::Do(iterate_domain_evaluator, IntervalType());
                (this->m_iterate_domain).advance_ij<1>(-range_t::jminus::value, 0);
                printf("END JMINUS %d %d %d %d \n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
            }
            else if(range_t::jplus::value != 0 && (threadIdx.y < -range_t::jminus::value + range_t::jplus::value))
            {
                const int joffset = range_t::jminus::value + cuda_block_size_j;
                printf("JPlUS %d %d %d %d %d \n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, joffset);

                (this->m_iterate_domain).advance_ij<1>(joffset, 0);
                functor_t::Do(iterate_domain_evaluator, IntervalType());
                (this->m_iterate_domain).advance_ij<1>(-joffset, 0);
                printf("END JPlUS %d %d %d %d %d \n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, joffset);
            }
            else if(range_t::iminus::value != 0 && (threadIdx.y < -range_t::jminus::value + range_t::jplus::value + 1))
            {
                printf("IMINUS %d %d %d %d \n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

                const int ioffset = -threadIdx.x - (threadIdx.x % (-range_t::iminus::value))-1;
                const int joffset = -threadIdx.y + (threadIdx.x / (-range_t::iminus::value) );

                printf("IMINS %d %d %d %d \n", ioffset, joffset, threadIdx.y, cuda_block_size_j);

                if(joffset + threadIdx.y < cuda_block_size_j)
                {
                    printf("IONS\n");
                    (this->m_iterate_domain).advance_ij<0>(ioffset, 0);
                    (this->m_iterate_domain).advance_ij<1>(joffset, 0);
                    functor_t::Do(iterate_domain_evaluator, IntervalType());
                    (this->m_iterate_domain).advance_ij<0>(-ioffset, 0);
                    (this->m_iterate_domain).advance_ij<1>(-joffset, 0);
                }
                printf("END IMINUS %d %d %d %d \n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
            }
            else if(range_t::iplus::value != 0 && (threadIdx.y < -range_t::jminus::value + range_t::jplus::value +
                    (range_t::iminus::value != 0 ? 1 : 0) + 1))
            {
                printf("IPLUS %d %d %d %d \n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

                const int ioffset = blockDim.x + (threadIdx.x % (range_t::iplus::value));
                const int joffset = -threadIdx.y + (threadIdx.x / (range_t::iplus::value) );

                if(joffset + threadIdx.y < cuda_block_size_j)
                {
                    (this->m_iterate_domain).advance_ij<0>(ioffset, 0);
                    (this->m_iterate_domain).advance_ij<1>(joffset, 0);
                    functor_t::Do(iterate_domain_evaluator, IntervalType());
                    (this->m_iterate_domain).advance_ij<0>(-ioffset, 0);
                    (this->m_iterate_domain).advance_ij<1>(-joffset, 0);
                }
                printf("END IPLUS %d %d %d %d \n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
            }
            else if(range_t::iplus::value != 0 || range_t::iminus::value != 0)
            {
//                const int joffset = block_size_t::j_size_t::value + range_t::jminus::value - range_t::jplus::value -
//                        (range_t::iminus::value != 0 ? 1 : 0) - (range_t::iplus::value != 0 ? 1 : 0);
                const int joffset = blockDim.y + range_t::jminus::value - range_t::jplus::value -
                        (range_t::iminus::value != 0 ? 1 : 0) - (range_t::iplus::value != 0 ? 1 : 0);
                printf("MYOFF %d %d %d \n", joffset, threadIdx.x, threadIdx.y);

                if(threadIdx.y + joffset < backend_traits_from_id<RunFunctorArguments::backend_id_t::value>::block_size_t::j_size_t::value)
                {
                    (this->m_iterate_domain).advance_ij<1>(joffset, 0);
                    functor_t::Do(iterate_domain_evaluator, IntervalType());
                    (this->m_iterate_domain).advance_ij<1>(-joffset, 0);
                }
            }
            __syncthreads();
        }

    };
}
