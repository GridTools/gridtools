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
#include <iostream>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/for_each.hpp>
#include "stencil-composition/level.hpp"
#include "stencil-composition/interval.hpp"
#include "stencil-composition/functor_do_methods.hpp"

using namespace gridtools;

// test functor 1
struct Functor0
{
    template <typename TArguments>
    static void Do(TArguments const& args, interval<level<3,-1>, level<3,-1> >)
    {
        std::cout << "Functor0:Do(Interval<Level<3,-1>, Level<3,-1> >) called" << std::endl;
    }
};

// test functor 1
struct Functor1
{
    template <typename TArguments>
    static void Do(TArguments const& args, interval<level<0,1>, level<2,-1> >)
    {
        std::cout << "Functor1:Do(Interval<Level<0,1>, Level<2,-1> >) called" << std::endl;
    }
};

// test functor 2
struct Functor2
{
    template <typename TArguments>
    static void Do(TArguments const& args, interval<level<0,1>, level<1,-1> >)
    {
        std::cout << "Functor2:Do(Interval<Level<0,1>, Level<1,-1> >) called" << std::endl;
    }

    template <typename TArguments>
    static void Do(TArguments const& args, interval<level<1,1>, level<3,-1> >)
    {
        std::cout << "Functor2:Do(Interval<Level<1,1>, Level<3,-1> >) called" << std::endl;
    }
};

// illegal functor
struct IllegalFunctor
{
    template <typename TArguments>
    static void Do(TArguments const& args, interval<level<1,1>, level<2,-1> >) {}
    template <typename TArguments>
    static void Do(TArguments const& args, interval<level<1,1>, level<3,-2> >) {}
    template <typename TArguments>
    static void Do(TArguments const& args, interval<level<3,-1>, level<3,-1> >) {}
};

// functor printing level and index
struct PrintLevel
{
    template<typename T>
    void operator()(T)
    {
        typedef typename index_to_level<T>::type Level;
        typedef typename level_to_index<Level>::type Index;

        std::cout
            << "Index: " << Index::value << "\t"
            << "Level(" << Level::Splitter::value << ", " << Level::Offset::value << ")" << std::endl;
    }
};

// helper executing a functor via boost for each
template<typename TFunctor>
struct RunnerFunctor
{
    template<typename T>
    void operator()(T)
    {
        // define the do method interval type
        typedef typename make_interval<
            typename boost::mpl::first<T>::type,
            typename boost::mpl::second<T>::type
        >::type interval;

        int argument = 0;
        TFunctor::Do(argument, interval());
    }
};

// test method computing functor do methods
int main(int argc, char *argv[])
{
    std::cout
        << "Functor Do Methods" << std::endl
        << "==================" << std::endl;

    // test the level to index conversions be enumerating all levels in an range
    // (for test purposes convert the range into levels and back into an index)
    std::cout << "Verify the level index computation:" << std::endl;
    boost::mpl::for_each<
        boost::mpl::range_c<int, 0, 20>
    >(PrintLevel());
    std::cout << "Done!" << std::endl;

    // // check has_do_simple on a few examples
    BOOST_STATIC_ASSERT((has_do<Functor0,
                         make_interval<
                         level_to_index<level<3,-1> >::type,
                         level_to_index<level<3,-1> >::type
                         >::type >::value));
    BOOST_STATIC_ASSERT((has_do<Functor0,
                         level<3,-1>
                         >::value));
    BOOST_STATIC_ASSERT((!has_do<Functor0,
                         level<3,-2>
                         >::value));
    BOOST_STATIC_ASSERT((has_do<Functor0,
                         index_to_level<static_int<20> >::type
                         >::value));
    BOOST_STATIC_ASSERT((!has_do<Functor0,
                         index_to_level<static_int<21> >::type
                         >::value));
    // typedef index_to_level<static_int<20> >::type sss; // used these lines to find out the level index
    // typedef sss::ciao ciccio;                          // by trial and error
    BOOST_STATIC_ASSERT((!has_do<Functor0,
                         make_interval<
                         level_to_index<level<1,-1> >::type,
                         level_to_index<level<3,3> >::type
                         >::type >::value));


    // define the axis search interval
    typedef interval<level<0,-3>, level<3,3> > AxisInterval;

    // run all methods of functor 0
    std::cout << "Print Functor0 Do methods:" << std::endl;
    boost::mpl::for_each<
        compute_functor_do_methods<
            Functor0,
            AxisInterval
        >::type
    >(RunnerFunctor<Functor0>());
    std::cout << "Done!" << std::endl;

    // run all methods of functor 1
    std::cout << "Print Functor1 Do methods:" << std::endl;
    boost::mpl::for_each<
        compute_functor_do_methods<
            Functor1,
            AxisInterval
        >::type
    >(RunnerFunctor<Functor1>());
    std::cout << "Done!" << std::endl;

    // run all methods of functor 2
    std::cout << "Print Functor2 Do methods:" << std::endl;
    boost::mpl::for_each<
        compute_functor_do_methods<
            Functor2,
            AxisInterval
        >::type
    >(RunnerFunctor<Functor2>());
    std::cout << "Done!" << std::endl;

    // test illegal functor
    //typedef compute_functor_do_methods<IllegalFunctor, AxisInterval>::type TestIllegalFunctor;

    return 0;
}
