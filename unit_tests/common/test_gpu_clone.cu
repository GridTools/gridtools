/** This code tests a solution to have clone objects on GPUs. The objects can have references to
    data members that must be initialized on GPU with references on the device

    Authors: Mauro Bianco, Ugo Varetto

    This version uses a gpu enabled boost::fusion library
*/

#include "gtest/gtest.h"

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#define BOOST_NO_CXX11_RVALUE_REFERENCES
#endif

#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/zip_view.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/at.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "common/defs.hpp"
#include "common/gpu_clone.hpp"

using gridtools::uint_t;
using gridtools::int_t;

namespace gpu_clone_test {

    /********************************************************
    GENERIC CODE THAW WORKS WITH ANY (almost POD) OBJECT
    *********************************************************/

    /********************************************************
    SPECIFIC CODE WITH AN OBJECT THAT HAS REFERENCES
    BUT NEED TO BE CLONED ON GPU
    *********************************************************/

    struct A : public gridtools::clonable_to_gpu< A > {
        typedef boost::fusion::vector< int, double > v_type;
        v_type v1;
        v_type v2;

        typedef boost::fusion::vector< v_type &, v_type & > support_t;
        typedef boost::fusion::zip_view< support_t > zip_view_t;

        zip_view_t zip_view;

        A(v_type const &a, v_type const &b) : v1(a), v2(b), zip_view(support_t(v1, v2)) {}

        __host__ __device__ A(A const &a) : v1(a.v1), v2(a.v2), zip_view(support_t(v1, v2)) {}

        ~A() {}

        void update_gpu_copy() const { clone_to_device(); }

        __host__ __device__ void out() const {
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
            __host__ __device__ void operator()(int u) const { printf("%d, ", u); }

            __host__ __device__ void operator()(double u) const { printf("%e, ", u); }
        };

        struct print_zip {
            template < typename V >
            __host__ __device__ void operator()(V const &v) const {
                boost::fusion::for_each(v, print_elements());
                printf("\n");
            }
        };
    };

    /** class to test gpu_clonable data-members
     */
    struct B : public gridtools::clonable_to_gpu< B > {
        A a;

        B(typename A::v_type const &v1, typename A::v_type const &v2) : a(v1, v2) {
            //        clone_to_device();
        }

        __device__ __host__ B(B const &b) : a(b.a) {}

        __host__ __device__ void out() const { a.out(); }
    };

    struct mul2_f {
        template < typename U >
        __host__ __device__ void operator()(U &u) const {
            u *= 2;
        }
    };

    struct mul2_fz {
        template < typename U >
        __host__ __device__ void operator()(U const &u) const {
            boost::fusion::at_c< 0 >(u) *= 2;
            boost::fusion::at_c< 1 >(u) *= 2;
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
        template < typename T >
        __host__ __device__ // Avoid warning
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
        mul2<<<1,1>>>(a1.gpu_object_ptr);
        // clang-format on
        a1.clone_from_device();

        boost::fusion::for_each(a2.v1, mul2_f());
        boost::fusion::for_each(a2.v2, mul2_f());

        bool equal = true;
        if (boost::fusion::at_c< 0 >(a1.v1) != boost::fusion::at_c< 0 >(a2.v1))
            equal = false;
        if (boost::fusion::at_c< 1 >(a1.v1) != boost::fusion::at_c< 1 >(a2.v1))
            equal = false;
        if (boost::fusion::at_c< 0 >(a1.v2) != boost::fusion::at_c< 0 >(a2.v2))
            equal = false;
        if (boost::fusion::at_c< 1 >(a1.v2) != boost::fusion::at_c< 1 >(a2.v2))
            equal = false;

        typename A::v_type bw1(m * 8, m * 1.23456789);
        typename A::v_type bw2(m * 7, m * 9.87654321);

        B b1(bw1, bw2);
        B b2(bw1, bw2);

        // b.out();

        // printf("Now doing the same on GPU");

        b1.clone_to_device();
        // clang-format off
        minus1<<<1,1>>>(b1.gpu_object_ptr);
        // clang-format on
        b1.clone_from_device();

        boost::fusion::for_each(b2.a.v1, minus1_f());
        boost::fusion::for_each(b2.a.v2, minus1_f());

        if (boost::fusion::at_c< 0 >(b1.a.v1) != boost::fusion::at_c< 0 >(b2.a.v1))
            equal = false;
        if (boost::fusion::at_c< 1 >(b1.a.v1) != boost::fusion::at_c< 1 >(b2.a.v1))
            equal = false;
        if (boost::fusion::at_c< 0 >(b1.a.v2) != boost::fusion::at_c< 0 >(b2.a.v2))
            equal = false;
        if (boost::fusion::at_c< 1 >(b1.a.v2) != boost::fusion::at_c< 1 >(b2.a.v2))
            equal = false;

        return equal;
    }
} // namespace gpu_clone_test

TEST(test_gpu_clone, test_gpu_clone) { EXPECT_EQ(gpu_clone_test::test_gpu_clone(), true); }
