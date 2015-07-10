#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#include <iostream>
#include <common/layout_map.hpp>

#ifndef NDEBUG
#define _output(x)         std::cout << x << std::endl;
#else
#define _output(x)
#endif

bool test_layout_map () {

    bool success = true;

    if (gridtools::layout_map<2>::at<0>() == 2) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3>::at<0>() == 1) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3>::at<1>() == 3) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3,-3>::at<0>() == 1) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3,-3>::at<1>() == 3) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3,-3>::at<2>() == -3) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3,-3,5>::at<0>() == 1) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3,-3,5>::at<1>() == 3) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3,-3,5>::at<2>() == -3) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3,-3,5>::at<3>() == 5) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    ////////////////////////////////////////////////////////////////////
    if (gridtools::layout_map<2>()[0] == 2) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3>()[0] == 1) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3>()[1] == 3) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3,-3>()[0] == 1) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3,-3>()[1] == 3) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3,-3>()[2] == -3) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3,-3,5>()[0] == 1) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3,-3,5>()[1] == 3) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3,-3,5>()[2] == -3) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,3,-3,5>()[3] == 5) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }

    typedef gridtools::layout_transform<gridtools::layout_map<0,1>, gridtools::layout_map<0,1> >::type transf0;

    if (transf0::at<0>() == 0) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (transf0::at<1>() == 1) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    typedef gridtools::layout_transform<gridtools::layout_map<0,1>, gridtools::layout_map<1,0> >::type transf01;

    if (transf01::at<0>() == 1) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (transf01::at<1>() == 0) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    typedef gridtools::layout_transform<gridtools::layout_map<1,0>, gridtools::layout_map<1,0> >::type transf02;

    if (transf02::at<0>() == 0) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (transf02::at<1>() == 1) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    typedef gridtools::layout_transform<gridtools::layout_map<2,0,1>, gridtools::layout_map<2,1,0> >::type transf;

    if (transf::at<0>() == 1) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (transf::at<1>() == 0) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (transf::at<2>() == 2) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    typedef gridtools::layout_transform<gridtools::layout_map<1,2,0>, gridtools::layout_map<0,1,2> >::type transf2;

    if (transf2::at<0>() == 1) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (transf2::at<1>() == 2) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (transf2::at<2>() == 0 ) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    int a=10,b=100,c=1000;

    if (gridtools::layout_map<2,0,1>::select<0>(a,b,c) == c) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<2,0,1>::select<1>(a,b,c) == a) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<2,0,1>::select<2>(a,b,c) == b) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,2,0>::select<0>(a,b,c) == b) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,2,0>::select<1>(a,b,c) == c) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<1,2,0>::select<2>(a,b,c) == a) {
        _output("PASSED");
    } else {
        _output("FAILED"); success = false;
    }
    if (gridtools::layout_map<2,0,1>::find<0>(a,b,c) == b) {
        _output("PASSED");
    } else {
        _output("FAILED "); success = false;
    }
    if (gridtools::layout_map<2,0,1>::find<1>(a,b,c) == c) {
        _output("PASSED");
    } else {
        _output("FAILED "); success = false;
    }
    if (gridtools::layout_map<2,0,1>::find<2>(a,b,c) == a) {
        _output("PASSED");
    } else {
        _output("FAILED "); success = false;
    }

    ////// TESTING FIND_VAL
    if (gridtools::layout_map<2,0,1>::find_val<0,int,666>(a,b,c) == b) {
        _output("PASSED");
    } else {
        _output("FAILED "); success = false;
    }
    if (gridtools::layout_map<2,0,1>::find_val<1,int,666>(a,b,c) == c) {
        _output("PASSED");
    } else {
        _output("FAILED "); success = false;
    }
    if (gridtools::layout_map<2,0,1>::find_val<2,int,666>(a,b,c) == a) {
        _output("PASSED");
    } else {
        _output("FAILED "); success = false;
    }
    if (gridtools::layout_map<2,0,1>::find_val<3,int,666>(a,b,c) == 666) {
        _output("PASSED");
    } else {
        _output("FAILED "); success = false;
    }
    return success;
}


#ifndef SILENT_RUN
int main() {
    return test_layout_map();
}
#endif
