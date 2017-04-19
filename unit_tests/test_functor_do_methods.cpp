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
#include <iostream>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/for_each.hpp>
#include "stencil-composition/level.hpp"
#include "stencil-composition/interval.hpp"
#include "stencil-composition/functor_do_methods.hpp"

using namespace gridtools;

// test functor 1
struct Functor0 {
    template < typename TArguments >
    static void Do(TArguments const &args, interval< level< 3, -1 >, level< 3, -1 > >) {
        std::cout << "Functor0:Do(Interval<Level<3,-1>, Level<3,-1> >) called" << std::endl;
    }
};

// test functor 1
struct Functor1 {
    template < typename TArguments >
    static void Do(TArguments const &args, interval< level< 0, 1 >, level< 2, -1 > >) {
        std::cout << "Functor1:Do(Interval<Level<0,1>, Level<2,-1> >) called" << std::endl;
    }
};

// test functor 2
struct Functor2 {
    template < typename TArguments >
    static void Do(TArguments const &args, interval< level< 0, 1 >, level< 1, -1 > >) {
        std::cout << "Functor2:Do(Interval<Level<0,1>, Level<1,-1> >) called" << std::endl;
    }

    template < typename TArguments >
    static void Do(TArguments const &args, interval< level< 1, 1 >, level< 3, -1 > >) {
        std::cout << "Functor2:Do(Interval<Level<1,1>, Level<3,-1> >) called" << std::endl;
    }
};

// illegal functor
struct IllegalFunctor {
    template < typename TArguments >
    static void Do(TArguments const &args, interval< level< 1, 1 >, level< 2, -1 > >) {}
    template < typename TArguments >
    static void Do(TArguments const &args, interval< level< 1, 1 >, level< 3, -2 > >) {}
    template < typename TArguments >
    static void Do(TArguments const &args, interval< level< 3, -1 >, level< 3, -1 > >) {}
};

// functor printing level and index
struct PrintLevel {
    template < typename T >
    void operator()(T) {
        typedef typename index_to_level< T >::type Level;
        typedef typename level_to_index< Level >::type Index;

        std::cout << "Index: " << Index::value << "\t"
                  << "Level(" << Level::Splitter::value << ", " << Level::Offset::value << ")" << std::endl;
    }
};

// helper executing a functor via boost for each
template < typename TFunctor >
struct RunnerFunctor {
    template < typename T >
    void operator()(T) {
        // define the do method interval type
        typedef typename make_interval< typename boost::mpl::first< T >::type,
            typename boost::mpl::second< T >::type >::type interval;

        int argument = 0;
        TFunctor::Do(argument, interval());
    }
};

// test method computing functor do methods
int main(int argc, char *argv[]) {
    std::cout << "Functor Do Methods" << std::endl << "==================" << std::endl;

    // test the level to index conversions be enumerating all levels in an range
    // (for test purposes convert the range into levels and back into an index)
    std::cout << "Verify the level index computation:" << std::endl;
    boost::mpl::for_each< boost::mpl::range_c< int, 0, 20 > >(PrintLevel());
    std::cout << "Done!" << std::endl;

    // // check has_do_simple on a few examples
    BOOST_STATIC_ASSERT((has_do< Functor0,
        make_interval< level_to_index< level< 3, -1 > >::type,
                                     level_to_index< level< 3, -1 > >::type >::type >::value));
    BOOST_STATIC_ASSERT((has_do< Functor0, level< 3, -1 > >::value));
    BOOST_STATIC_ASSERT((!has_do< Functor0, level< 3, -2 > >::value));
    BOOST_STATIC_ASSERT((has_do< Functor0, index_to_level< static_int< 20 > >::type >::value));
    BOOST_STATIC_ASSERT((!has_do< Functor0, index_to_level< static_int< 21 > >::type >::value));
    // typedef index_to_level<static_int<20> >::type sss; // used these lines to find out the level index
    // typedef sss::ciao ciccio;                          // by trial and error
    BOOST_STATIC_ASSERT((!has_do< Functor0,
                         make_interval< level_to_index< level< 1, -1 > >::type,
                                      level_to_index< level< 3, 3 > >::type >::type >::value));

    // define the axis search interval
    typedef interval< level< 0, -3 >, level< 3, 3 > > AxisInterval;

    // run all methods of functor 0
    std::cout << "Print Functor0 Do methods:" << std::endl;
    boost::mpl::for_each< compute_functor_do_methods< Functor0, AxisInterval >::type >(RunnerFunctor< Functor0 >());
    std::cout << "Done!" << std::endl;

    // run all methods of functor 1
    std::cout << "Print Functor1 Do methods:" << std::endl;
    boost::mpl::for_each< compute_functor_do_methods< Functor1, AxisInterval >::type >(RunnerFunctor< Functor1 >());
    std::cout << "Done!" << std::endl;

    // run all methods of functor 2
    std::cout << "Print Functor2 Do methods:" << std::endl;
    boost::mpl::for_each< compute_functor_do_methods< Functor2, AxisInterval >::type >(RunnerFunctor< Functor2 >());
    std::cout << "Done!" << std::endl;

    // test illegal functor
    // typedef compute_functor_do_methods<IllegalFunctor, AxisInterval>::type TestIllegalFunctor;

    return 0;
}
