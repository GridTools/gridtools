/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/conditionals/condition_pool.hpp>

namespace test_conditional_switches {
    using namespace gridtools;
    using namespace enumtype;

#define BACKEND_BLOCK

#ifdef CUDA_EXAMPLE
#define BACKEND backend< enumtype::Cuda, enumtype::GRIDBACKEND, enumtype::Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< enumtype::Host, enumtype::GRIDBACKEND, enumtype::Block >
#else
#define BACKEND backend< enumtype::Host, enumtype::GRIDBACKEND, enumtype::Naive >
#endif
#endif

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    template < uint_t Id >
    struct functor1 {

        typedef accessor< 0, enumtype::inout > p_dummy;
        typedef accessor< 1, enumtype::inout > p_dummy_tmp;

        typedef boost::mpl::vector2< p_dummy, p_dummy_tmp > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(p_dummy()) += Id;
        }
    };

    template < uint_t Id >
    struct functor2 {

        typedef accessor< 0, enumtype::inout > p_dummy;
        typedef accessor< 1, enumtype::in > p_dummy_tmp;

        typedef boost::mpl::vector2< p_dummy, p_dummy_tmp > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(p_dummy()) += Id;
        }
    };

    bool test() {

        bool p = true;
        auto cond_ = new_switch_variable([&p]() { return p ? 0 : 5; });
        auto nested_cond_ = new_switch_variable([]() { return 1; });
        auto other_cond_ = new_switch_variable([&p]() { return p ? 1 : 2; });

        grid< axis > grid_({0, 0, 0, 6, 7}, {0, 0, 0, 6, 7});
        grid_.value_list[0] = 0;
        grid_.value_list[1] = 7;

        typedef gridtools::layout_map< 2, 1, 0 > layout_t; // stride 1 on i
        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_data_t;
        typedef BACKEND::storage_info< __COUNTER__, layout_t > tmp_meta_data_t;
<<<<<<< HEAD
        typedef BACKEND::storage_type< float_type, meta_data_t >::type storage_t;
        typedef BACKEND::temporary_storage_type< float_type, tmp_meta_data_t >::type tmp_storage_t;

        meta_data_t meta_data_(8, 8, 8);
        storage_t dummy(meta_data_, 0., "dummy");
=======
        typedef BACKEND::storage_type< uint_t, meta_data_t >::type storage_t;
        typedef BACKEND::temporary_storage_type< uint_t, tmp_meta_data_t >::type tmp_storage_t;

        meta_data_t meta_data_(8, 8, 8);
        storage_t dummy(meta_data_, 0, "dummy");
>>>>>>> master
        typedef arg< 0, storage_t > p_dummy;
        typedef arg< 1, tmp_storage_t > p_dummy_tmp;

        typedef boost::mpl::vector2< p_dummy, p_dummy_tmp > arg_list;
        aggregator_type< arg_list > domain_(boost::fusion::make_vector(&dummy));

        auto comp_ = make_computation< BACKEND >(
            domain_,
            grid_,
            make_multistage(enumtype::execute< enumtype::forward >(),
                make_stage< functor1< 0 > >(p_dummy(), p_dummy_tmp()),
                make_stage< functor2< 0 > >(p_dummy(), p_dummy_tmp())),
            switch_(cond_,
                case_(0,
                        make_multistage(enumtype::execute< enumtype::forward >(),
                            make_stage< functor1< 1 > >(p_dummy(), p_dummy_tmp()),
                            make_stage< functor2< 1 > >(p_dummy(), p_dummy_tmp()))),
                // case_(1,
                //         make_multistage(enumtype::execute< enumtype::forward >(),
                //             make_stage< functor1< 2 > >(p_dummy(), p_dummy_tmp()),
                //             make_stage< functor2< 2 > >(p_dummy(), p_dummy_tmp()))),
                // case_(2,
                //         make_multistage(enumtype::execute< enumtype::forward >(),
                //             make_stage< functor1< 3 > >(p_dummy(), p_dummy_tmp()),
                //             make_stage< functor2< 3 > >(p_dummy(), p_dummy_tmp()))),
                // case_(3,
                //         make_multistage(enumtype::execute< enumtype::forward >(),
                //             make_stage< functor1< 4 > >(p_dummy(), p_dummy_tmp()),
                //             make_stage< functor2< 4 > >(p_dummy(), p_dummy_tmp()))),
                // case_(4,
                //         make_multistage(enumtype::execute< enumtype::forward >(),
                //             make_stage< functor1< 5 > >(p_dummy(), p_dummy_tmp()),
                //             make_stage< functor2< 5 > >(p_dummy(), p_dummy_tmp()))),
                case_(5,
                        switch_(nested_cond_,
                            // case_(2,
                            //         make_multistage(enumtype::execute< enumtype::forward >(),
                            //             make_stage< functor1< 1000 > >(
                            //                      p_dummy(), p_dummy_tmp()),
                            //             make_stage< functor2< 1000 > >(
                            //                      p_dummy(), p_dummy_tmp()))),
                            case_(1,
                                    make_multistage(enumtype::execute< enumtype::forward >(),
                                        make_stage< functor1< 2000 > >(p_dummy(), p_dummy_tmp()),
                                        make_stage< functor2< 2000 > >(p_dummy(), p_dummy_tmp()))),
                            default_(make_multistage(enumtype::execute< enumtype::forward >(),
                                make_stage< functor1< 3000 > >(p_dummy(), p_dummy_tmp()),
                                make_stage< functor2< 3000 > >(p_dummy(), p_dummy_tmp()))))),
                // case_(6,
                //         make_multistage(enumtype::execute< enumtype::forward >(),
                //             make_stage< functor1< 6 > >(p_dummy(), p_dummy_tmp()),
                //             make_stage< functor2< 6 > >(p_dummy(), p_dummy_tmp()))),
                default_(make_multistage(enumtype::execute< enumtype::forward >(),
                    make_stage< functor1< 7 > >(p_dummy(), p_dummy_tmp()),
                    make_stage< functor2< 7 > >(p_dummy(), p_dummy_tmp())))),
            switch_(other_cond_,
                case_(2,
                        make_multistage(enumtype::execute< enumtype::forward >(),
                            make_stage< functor1< 10 > >(p_dummy(), p_dummy_tmp()),
                            make_stage< functor2< 10 > >(p_dummy(), p_dummy_tmp()))),
                case_(1,
                        make_multistage(enumtype::execute< enumtype::forward >(),
                            make_stage< functor1< 20 > >(p_dummy(), p_dummy_tmp()),
                            make_stage< functor2< 20 > >(p_dummy(), p_dummy_tmp()))),
                default_(make_multistage(enumtype::execute< enumtype::forward >(),
                    make_stage< functor1< 30 > >(p_dummy(), p_dummy_tmp()),
                    make_stage< functor2< 30 > >(p_dummy(), p_dummy_tmp())))),
            make_multistage(enumtype::execute< enumtype::forward >(),
                make_stage< functor1< 400 > >(p_dummy(), p_dummy_tmp()),
                make_stage< functor2< 400 > >(p_dummy(), p_dummy_tmp())));

        bool result = true;

        comp_->ready();
        comp_->steady();
        comp_->run();
#ifdef __CUDACC__
        dummy.d2h_update();
#endif
        result = result && (dummy(0, 0, 0) == 842);
<<<<<<< HEAD
=======
        std::cout << "dummy is: " << dummy(0, 0, 0) << std::endl;
>>>>>>> master

        p = false;
#ifdef __CUDACC__
        // this is necessary otherwise "finalize" will think that the copy device-to-host is not needed
        dummy.set_on_device();
#endif
        comp_->run();
        comp_->finalize();
        result = result && (dummy(0, 0, 0) == 5662);
<<<<<<< HEAD
=======
        std::cout << "dummy became: " << dummy(0, 0, 0) << std::endl;
>>>>>>> master

        return result;
    }
} // namespace test_conditional

TEST(stencil_composition, conditional_switch) { EXPECT_TRUE(test_conditional_switches::test()); }
