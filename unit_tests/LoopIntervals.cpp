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
#include <iostream>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/transform.hpp>
#include "stencil-composition/interval.hpp"
#include "stencil-composition/functor_do_methods.hpp"
#include "stencil-composition/loopintervals.hpp"

using namespace gridtools;

// test functor 1
struct Functor0
{
    template <typename TArguments>
    static void Do(TArguments const& args, interval<level<3,-1>, level<3,-1> >) {}
};

// test functor 1
struct Functor1
{
    template <typename TArguments>
    static void Do(TArguments const& args, interval<level<0,1>, level<2,-1> >) {}
};

// test functor 2
struct Functor2
{
    template <typename TArguments>
    static void Do(TArguments const& args, interval<level<0,1>, level<1,-1> >) {}

    template <typename TArguments>
    static void Do(TArguments const& args, interval<level<1,1>, level<3,-1> >) {}
};

// helper printing the loop index pairs
struct PrintIndexPairs
{
    template<typename TIndexPair>
    void operator()(TIndexPair)
    {
        // extract the level information
        typedef typename index_to_level<
            typename boost::mpl::first<TIndexPair>::type
        >::type FromLevel;
        typedef typename index_to_level<
            typename boost::mpl::second<TIndexPair>::type
        >::type ToLevel;

        std::cout
            << "(" << FromLevel::Splitter::value << "," << FromLevel::Offset::value << ") -> "
            << "(" << ToLevel::Splitter::value << "," << ToLevel::Offset::value << ")"
            << std::endl;
    }
};

// test method computing loop intervals
int main(int argc, char *argv[])
{
    std::cout
        << "Loop Intervals" << std::endl
        << "==============" << std::endl;

    // define the axis search interval
    typedef interval<level<0,-3>, level<3,3> > AxisInterval;

    // compute the functor do methods
    typedef boost::mpl::transform<
        boost::mpl::vector<Functor0, Functor1, Functor2>,
        compute_functor_do_methods<boost::mpl::_, AxisInterval>
    >::type FunctorDoMethods;

    std::cout << "Print the Functor0, Functor1 and Functor2 loop intervals:" << std::endl;
    boost::mpl::for_each<
        compute_loop_intervals<
            FunctorDoMethods,
            AxisInterval
        >::type
    >(PrintIndexPairs());
    std::cout << "Done!" << std::endl;

    return 0;
}
