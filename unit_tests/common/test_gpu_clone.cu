/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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
/** This code tests a solution to have clone objects on GPUs. The objects can have references to
    data members that must be initialized on GPU with references on the device

    Authors: Mauro Bianco, Ugo Varetto

    This version uses a gpu enabled boost::fusion library
*/

#include "gtest/gtest.h"

#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/zip_view.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/common/gpu_clone.hpp>

using gridtools::int_t;
using gridtools::uint_t;

namespace gpu_clone_test {

    /********************************************************
    GENERIC CODE THAW WORKS WITH ANY (almost POD) OBJECT
    *********************************************************/

    /********************************************************
    SPECIFIC CODE WITH AN OBJECT THAT HAS REFERENCES
    BUT NEED TO BE CLONED ON GPU
    *********************************************************/

    struct A : public gridtools::clonable_to_gpu<A> {
        typedef boost::fusion::vector<int, double> v_type;
        v_type v1;
        v_type v2;

        typedef boost::fusion::vector<v_type &, v_type &> support_t;
        typedef boost::fusion::zip_view<support_t> zip_view_t;

        zip_view_t zip_view;

        A(v_type const &a, v_type const &b) : v1(a), v2(b), zip_view(support_t(v1, v2)) {}

        GT_FUNCTION A(A const &a) : v1(a.v1), v2(a.v2), zip_view(support_t(v1, v2)) {}

        ~A() {}

        void update_gpu_copy() const { clone_to_device(); }

        GT_FUNCTION void out() const {
            printf("v1:  ");
            boost::fusion::for_each(v1, print_elements());
            printf("\n");

            printf("v2:  ");
            boost::fusion::for_each(v2, print_elements());
            printf("\n");

            printf("zip: ");
            boost::fusion::for_each(zip_view, print_zip());
            printf("\n");
        }

      private:
        struct print_elements {
            GT_FUNCTION void operator()(int u) const { printf("%d, ", u); }

            GT_FUNCTION void operator()(double u) const { printf("%e, ", u); }
        };

        struct print_zip {
            template <typename V>
            GT_FUNCTION void operator()(V const &v) const {
                boost::fusion::for_each(v, print_elements());
                printf("\n");
            }
        };
    };

    /** class to test gpu_clonable data-members
     */
    struct B : public gridtools::clonable_to_gpu<B> {
        A a;

        B(typename A::v_type const &v1, typename A::v_type const &v2) : a(v1, v2) {
            //        clone_to_device();
        }

        GT_FUNCTION B(B const &b) : a(b.a) {}

        GT_FUNCTION void out() const { a.out(); }
    };

    struct mul2_f {
        template <typename U>
        GT_FUNCTION void operator()(U &u) const {
            u *= 2;
        }
    };

    struct mul2_fz {
        template <typename U>
        GT_FUNCTION void operator()(U const &u) const {
            boost::fusion::at_c<0>(u) *= 2;
            boost::fusion::at_c<1>(u) *= 2;
        }
    };

    __global__ void mul2(A *a) { boost::fusion::for_each(a->zip_view, mul2_fz()); }

    // __global__
    // void print_on_gpu(A * a) {
    //     a->out();
    // }

    // __global__
    // void print_on_gpu(B * b) {
    //     b->out();
    // }

    struct minus1_f {
        template <typename T>
        GT_FUNCTION // Avoid warning
            void
            operator()(T &x) const {
            x -= 1;
        }
    };

    __global__ void minus1(B *b) {
        boost::fusion::for_each(b->a.v1, minus1_f());
        boost::fusion::for_each(b->a.v2, minus1_f());
    }

    bool test_gpu_clone() {

        uint_t m = 7;

        typename A::v_type w1(m * 1, m * 3.1415926);
        typename A::v_type w2(m * 2, m * 2.7182818);

        A a1(w1, w2);
        A a2(w1, w2);
        a1.update_gpu_copy();

        a2.update_gpu_copy();

        // clang-format off
        mul2< <<1,1> >>(a1.gpu_object_ptr);
        // clang-format on
        a1.clone_from_device();

        boost::fusion::for_each(a2.v1, mul2_f());
        boost::fusion::for_each(a2.v2, mul2_f());

        bool equal = true;
        if (boost::fusion::at_c<0>(a1.v1) != boost::fusion::at_c<0>(a2.v1))
            equal = false;
        if (boost::fusion::at_c<1>(a1.v1) != boost::fusion::at_c<1>(a2.v1))
            equal = false;
        if (boost::fusion::at_c<0>(a1.v2) != boost::fusion::at_c<0>(a2.v2))
            equal = false;
        if (boost::fusion::at_c<1>(a1.v2) != boost::fusion::at_c<1>(a2.v2))
            equal = false;

        typename A::v_type bw1(m * 8, m * 1.23456789);
        typename A::v_type bw2(m * 7, m * 9.87654321);

        B b1(bw1, bw2);
        B b2(bw1, bw2);

        // b.out();

        // printf("Now doing the same on GPU");

        b1.clone_to_device();
        // clang-format off
        minus1< <<1,1> >>(b1.gpu_object_ptr);
        // clang-format on
        b1.clone_from_device();

        boost::fusion::for_each(b2.a.v1, minus1_f());
        boost::fusion::for_each(b2.a.v2, minus1_f());

        if (boost::fusion::at_c<0>(b1.a.v1) != boost::fusion::at_c<0>(b2.a.v1))
            equal = false;
        if (boost::fusion::at_c<1>(b1.a.v1) != boost::fusion::at_c<1>(b2.a.v1))
            equal = false;
        if (boost::fusion::at_c<0>(b1.a.v2) != boost::fusion::at_c<0>(b2.a.v2))
            equal = false;
        if (boost::fusion::at_c<1>(b1.a.v2) != boost::fusion::at_c<1>(b2.a.v2))
            equal = false;

        return equal;
    }
} // namespace gpu_clone_test

TEST(test_gpu_clone, test_gpu_clone) { EXPECT_EQ(gpu_clone_test::test_gpu_clone(), true); }
