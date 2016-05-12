#include <iostream>

#include "range.hpp"
#include <boost/mpl/for_each.hpp>

using namespace gridtools;

typedef range<-1,1,-1,1> range0;
typedef range<-2,2,-2,2> range1;
typedef range<-3,3,-3,3> range2;

typedef boost::mpl::vector<range0, range1, range2> input;

struct print {
    template <typename T>
    void operator()(T const&) const {
        std::cout << T() << std::endl;
    }
};


int main() {

    std::cout << enclosing_range<range0, range1>::type() << std::endl;
    std::cout << enclosing_range<range1, range0>::type() << std::endl;

    std::cout << "input" << std::endl;

    boost::mpl::for_each<input>(print());

    std::cout << std::endl;
    std::cout << "output" << std::endl;

    boost::mpl::for_each<prefix_on_ranges<input>::type>(print());

    return 0;
}
