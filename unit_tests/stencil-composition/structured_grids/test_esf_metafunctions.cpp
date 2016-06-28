/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>

using namespace gridtools;
using namespace enumtype;

typedef interval< level< 0, -1 >, level< 1, -1 > > x_interval;
struct print_r {
    template <typename T>
    void operator()(T const& ) const {
        std::cout << typename T::first() << " " << typename T::second() << std::endl;
    }
};

struct functor0{
    typedef accessor< 0, enumtype::in, extent< 0, 0, -1, 3, -2, 0 > > in0;
    typedef accessor< 1, enumtype::in, extent< -1, 1, 0, 2, -1, 2 > > in1;
    typedef accessor< 2, enumtype::in, extent< -3, 3, -1, 2, 0, 1 > > in2;
    typedef accessor< 3, enumtype::inout > out;

    typedef boost::mpl::vector< in0, in1, in2, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {}
};

struct functor1 {
    typedef accessor< 0, enumtype::in, extent< 0, 1, -1, 2, 0, 0 > > in0;
    typedef accessor< 1, enumtype::inout > out;
    typedef accessor< 2, enumtype::in, extent< -3, 0, -3, 0, 0, 2 > > in2;
    typedef accessor< 3, enumtype::in, extent< 0, 2, 0, 2, -2, 3 > > in3;

    typedef boost::mpl::vector< in0, out, in2, in3 > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {}
};

struct functor2 {
    typedef accessor< 0, enumtype::in, extent< -3, 3, -1, 0, -2, 1 > > in0;
    typedef accessor< 1, enumtype::in, extent< -3, 1, -2, 1, 0, 2 > > in1;
    typedef accessor< 2, enumtype::inout > out;

    typedef boost::mpl::vector< in0, in1, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {}
};

struct functor3 {
    typedef accessor< 0, enumtype::in, extent< 0, 3, 0, 1, -2, 0 > > in0;
    typedef accessor< 1, enumtype::in, extent< -2, 3, 0, 2, -3, 1 > > in1;
    typedef accessor<2, enumtype::inout> out;
    typedef accessor< 3, enumtype::in, extent< -1, 3, -3, 0, -3, 2 > > in3;

    typedef boost::mpl::vector<in0,in1,out,in3> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {}
};

struct functor4 {
    typedef accessor< 0, enumtype::in, extent< 0, 3, -2, 1, -3, 2 > > in0;
    typedef accessor< 1, enumtype::in, extent< -2, 3, 0, 3, -3, 2 > > in1;
    typedef accessor< 2, enumtype::in, extent< -1, 1, 0, 3, 0, 3 > > in2;
    typedef accessor< 3, enumtype::inout > out;

    typedef boost::mpl::vector< in0, in1, in2, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {}
};

struct functor5 {
    typedef accessor< 0, enumtype::in, extent< -3, 1, -1, 2, -1, 1 > > in0;
    typedef accessor< 1, enumtype::in, extent< 0, 1, -2, 2, 0, 3 > > in1;
    typedef accessor< 2, enumtype::in, extent< 0, 2, 0, 3, -1, 2 > > in2;
    typedef accessor< 3, enumtype::inout > out;

    typedef boost::mpl::vector< in0, in1, in2, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {}
};

struct functor6 {
    typedef accessor< 0, enumtype::inout > out;
    typedef accessor< 1, enumtype::in, extent< 0, 3, -3, 2, 0, 0 > > in1;
    typedef accessor< 2, enumtype::in, extent< -3, 2, 0, 2, -1, 2 > > in2;
    typedef accessor< 3, enumtype::in, extent< -1, 0, -1, 0, -1, 3 > > in3;

    typedef boost::mpl::vector< out, in1, in2, in3 > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {}
};

std::ostream& operator<<(std::ostream& s, functor0) { return s << "functor0"; }
std::ostream &operator<<(std::ostream &s, functor1) { return s << "functor1"; }
std::ostream &operator<<(std::ostream &s, functor2) { return s << "functor2"; }
std::ostream &operator<<(std::ostream &s, functor3) { return s << "functor3"; }
std::ostream &operator<<(std::ostream &s, functor4) { return s << "functor4"; }
std::ostream &operator<<(std::ostream &s, functor5) { return s << "functor5"; }
std::ostream &operator<<(std::ostream &s, functor6) { return s << "functor6"; }
#define BACKEND backend< Host, GRIDBACKEND, Block >

typedef layout_map< 2, 1, 0 > layout_t;
typedef BACKEND::storage_info< 0, layout_t > storage_info_type;
typedef BACKEND::storage_type< float_type, storage_info_type >::type storage_type;

typedef arg<0, storage_type> o0;
typedef arg< 1, storage_type > o1;
typedef arg< 2, storage_type > o2;
typedef arg< 3, storage_type > o3;
typedef arg< 4, storage_type > o4;
typedef arg< 5, storage_type > o5;
typedef arg< 6, storage_type > o6;
typedef arg< 7, storage_type > in0;
typedef arg< 8, storage_type > in1;
typedef arg< 9, storage_type > in2;
typedef arg< 10, storage_type > in3;
int main() {
    typedef decltype(make_stage< functor0 >(in0(), in1(), in2(), o0())) functor0__;
    typedef decltype(make_stage< functor1 >(in3(), o1(), in0(), o0())) functor1__;
    typedef decltype(make_stage< functor2 >(o0(), o1(), o2())) functor2__;
    typedef decltype(make_stage< functor3 >(in1(), in2(), o3(), o2())) functor3__;
    typedef decltype(make_stage< functor4 >(o0(), o1(), o3(), o4())) functor4__;
    typedef decltype(make_stage< functor5 >(in3(), o4(), in0(), o5())) functor5__;
    typedef decltype(make_stage< functor6 >(o6(), o5(), in1(), in2())) functor6__;
    typedef decltype(make_multistage(execute< forward >(),
        functor0__(),
        functor1__(),
        functor2__(),
        functor3__(),
        functor4__(),
        functor5__(),
        functor6__())) mss_t;
    typedef boost::mpl::vector< o0, o1, o2, o3, o4, o5, o6, in0, in1, in2, in3 > placeholders;

    typedef compute_extents_of< init_map_of_extents< placeholders >::type, 1 >::for_mss< mss_t >::type final_map;
    std::cout << "FINAL" << std::endl;
    boost::mpl::for_each< final_map >(print_r());

    GRIDTOOLS_STATIC_ASSERT(
        (std::is_same< boost::mpl::at< final_map, o0 >::type, extent< -5, 11, -10, 10, -5, 13 > >::type::value),
        "o0 extent<-5, 11, -10, 10, -5, 13> ");
    GRIDTOOLS_STATIC_ASSERT(
        (std::is_same< boost::mpl::at< final_map, o1 >::type, extent< -5, 9, -10, 8, -3, 10 > >::type::value),
        "o1 extent<-5, 9, -10, 8, -3, 10> ");
    GRIDTOOLS_STATIC_ASSERT(
        (std::is_same< boost::mpl::at< final_map, o2 >::type, extent< -2, 8, -8, 7, -3, 8 > >::type::value),
        "o2 extent<-2, 8, -8, 7, -3, 8> ");
    GRIDTOOLS_STATIC_ASSERT(
        (std::is_same< boost::mpl::at< final_map, o3 >::type, extent< -1, 5, -5, 7, 0, 6 > >::type::value),
        "o3 extent<-1, 5, -5, 7, 0, 6> ");
    GRIDTOOLS_STATIC_ASSERT(
        (std::is_same< boost::mpl::at< final_map, o4 >::type, extent< 0, 4, -5, 4, 0, 3 > >::type::value),
        "o4 extent<0, 4, -5, 4, 0, 3> ");
    GRIDTOOLS_STATIC_ASSERT(
        (std::is_same< boost::mpl::at< final_map, o5 >::type, extent< 0, 3, -3, 2, 0, 0 > >::type::value),
        "o5 extent<0, 3, -3, 2, 0, 0> ");
    GRIDTOOLS_STATIC_ASSERT(
        (std::is_same< boost::mpl::at< final_map, o6 >::type, extent< 0, 0, 0, 0, 0, 0 > >::type::value),
        "o6 extent<0, 0, 0, 0, 0, 0> ");
    GRIDTOOLS_STATIC_ASSERT(
        (std::is_same< boost::mpl::at< final_map, in0 >::type, extent< -8, 11, -13, 13, -7, 13 > >::type::value),
        "in0 extent<-8, 11, -13, 13, -7, 13> ");
    GRIDTOOLS_STATIC_ASSERT(
        (std::is_same< boost::mpl::at< final_map, in1 >::type, extent< -6, 12, -10, 12, -6, 15 > >::type::value),
        "in1 extent<-6, 12, -10, 12, -6, 15> ");
    GRIDTOOLS_STATIC_ASSERT(
        (std::is_same< boost::mpl::at< final_map, in2 >::type, extent< -8, 14, -11, 12, -5, 14 > >::type::value),
        "in2 extent<-8, 14, -11, 12, -5, 14> ");
    GRIDTOOLS_STATIC_ASSERT(
        (std::is_same< boost::mpl::at< final_map, in3 >::type, extent< -5, 10, -11, 10, -3, 10 > >::type::value),
        "in3 extent<-5, 10, -11, 10, -3, 10> ");
    /* total placeholders (rounded to 10) _SIZE = 20*/
    return 0;
}
