#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/pair.hpp>
#include "gt_for_each/for_each.hpp"
#include <boost/mpl/transform.hpp>
#include <boost/mpl/at.hpp>
#include <iostream>
#include "common/host_device.hpp"
#include "stencil-composition/interval.hpp"
#include "stencil-composition/loopintervals.hpp"
#include "stencil-composition/functor_do_methods.hpp"
#include "stencil-composition/functor_do_method_lookup_maps.hpp"

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

// helper printing a do method lookup map entry
template<typename TDoMethodLookUpMap>
struct PrintLoopInterval
{
    template<typename TLoopInterval>
    void operator()(TLoopInterval)
    {
        typedef typename boost::mpl::at<
            TDoMethodLookUpMap,
            TLoopInterval
            >::type DoInterval;
        // print the loop interval
        typedef typename index_to_level<typename TLoopInterval::first>::type FromLevel;
        typedef typename index_to_level<typename TLoopInterval::second>::type ToLevel;
        std::cout
            << "Loop (" << FromLevel::Splitter::value << "," << ToLevel::Offset::value << ") - "
            << "(" << FromLevel::Splitter::value << "," << ToLevel::Offset::value << ")\t->\t";

        // print the do method
        printDoInterval(static_cast<DoInterval*>(0));
    }

private:
    // either print void or the do method interval
    void printDoInterval(boost::mpl::void_*)
    {
        std::cout << "idle!" << std::endl;
    }
    template<typename TInterval>
    void printDoInterval(TInterval*)
    {
        std::cout
            << "Do (" << TInterval::FromLevel::Splitter::value << "," << TInterval::FromLevel::Offset::value << ") - "
            << "(" << TInterval::ToLevel::Splitter::value << "," << TInterval::ToLevel::Offset::value << ")" << std::endl;
    }
};

// helper printing the do method lookup map
template<
    typename TFunctors,
    typename TLoopIntervals,
    typename TFunctorDoMethodLookupMaps>
struct PrintDoMethodLookupMap
{
    template<typename TIndex>
    void operator()(TIndex)
    {

        typedef typename boost::mpl::at<TFunctors, TIndex>::type Functor;
        typedef typename boost::mpl::at<TFunctorDoMethodLookupMaps, static_int<0> >::type DoMethodLookUpMap;

        // print the functor name
        if(boost::is_same<Functor0, Functor>::value)
        {
            std::cout << "Functor0:" << std::endl;
        }
        else if(boost::is_same<Functor1, Functor>::value)
        {
            std::cout << "Functor1:" << std::endl;
        }
        else
        {
            std::cout << "Functor2:" << std::endl;
        }

        // print the map
        gridtools::for_each<
            TLoopIntervals
        >(PrintLoopInterval<DoMethodLookUpMap>());
    }
};

// test method computing do method lookup maps
int main(int argc, char *argv[])
{
#if defined(CXX11_ENABLED) && (__CUDA_ARCH__<=350)
    std::cout
        << "Functor Do Method Lookup Map" << std::endl
        << "============================" << std::endl;

    // define the axis search interval
    typedef interval<level<0,-3>, level<3,3> > AxisInterval;

    // define the functors
    typedef boost::mpl::vector<Functor0, Functor1, Functor2> Functors;

    // compute the functor do methods
    typedef boost::mpl::transform<
        Functors,
        compute_functor_do_methods<boost::mpl::_, AxisInterval>
    >::type FunctorsDoMethods;

    // compute the loop intervals
    typedef compute_loop_intervals<
        FunctorsDoMethods,
        AxisInterval
    >::type LoopIntervals;

    // compute the functor do method lookup maps
    typedef boost::mpl::transform<
        FunctorsDoMethods,
        compute_functor_do_method_lookup_map<boost::mpl::_, LoopIntervals>
    >::type FunctorDoMethodLookupMaps;

    // print all loop intervals of functor 0, 1 and 2
    std::cout << "Print the Functor0, Functor1 and Functor2 do method lookup maps:" << std::endl;
    gridtools::for_each<
        boost::mpl::range_c<uint_t, 0, boost::mpl::size<FunctorsDoMethods>::value>//from 0 to 2
    >(PrintDoMethodLookupMap<Functors, LoopIntervals, FunctorDoMethodLookupMaps>());
    std::cout << "Done!" << std::endl;
#endif
    return 0;
}
