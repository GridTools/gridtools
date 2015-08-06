#include <iostream>
#include <gt_for_each/for_each.hpp>
#include <boost/mpl/transform.hpp>
#include "stencil-composition/interval.hpp"
#include "stencil-composition/functor_do_methods.hpp"
#include "stencil-composition/loopintervals.hpp"

using namespace gridtools;

// test functor 1
struct Functor0
{
    template <typename TArguments>
    static void Do(TArguments& args, interval<level<3,-1>, level<3,-1> >) {}
};

// test functor 1
struct Functor1
{
    template <typename TArguments>
    static void Do(TArguments& args, interval<level<0,1>, level<2,-1> >) {}
};

// test functor 2
struct Functor2
{
    template <typename TArguments>
    static void Do(TArguments& args, interval<level<0,1>, level<1,-1> >) {}

    template <typename TArguments>
    static void Do(TArguments& args, interval<level<1,1>, level<3,-1> >) {}
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
    gridtools::for_each<
        compute_loop_intervals<
            FunctorDoMethods,
            AxisInterval
        >::type
    >(PrintIndexPairs());
    std::cout << "Done!" << std::endl;

    return 0;
}
