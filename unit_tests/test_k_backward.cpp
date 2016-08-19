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
#include <stencil-composition/execution_policy.hpp>
using namespace gridtools;
using namespace enumtype;


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

struct fake_domain{void increment(){}};
struct fake_coords{};

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
        typedef typename boost::mpl::at<TFunctorDoMethodLookupMaps, TIndex>::type DoMethodLookUpMap;

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

    struct extra_arguments{
        typedef Functor functor_t;
        typedef DoMethodLookupMap interval_map_t;
        typedef fake_domain local_domain_t;
        typedef fake_coords coords_t;};

    gridtools::for_each< TLoopIntervals >
        (_impl::run_f_on_interval
         <
         execution_type_t,
         extra_arguments
         >
         (f->m_domain,f->m_coords)
            );

    }
};


int main()
{
    // Definition of the physical dimensions of the problem.
    // The constructor takes the horizontal plane dimensions,
    // while the vertical ones are set according the the axis property soon after
    // gridtools::coordinates<axis> coords(2,d1-2,2,d2-2);
    int_t di[5] = {2, 2, 2, d1-2, d1};
    int_t dj[5] = {2, 2, 2, d2-2, d2};
    gridtools::coordinates<axis> coords(di, dj);
    coords.value_list[0] = 0;
    coords.value_list[1] = d3;

    std::cout
        << "Forward, Backward, and Parallel k loop iterations" << std::endl
        << "=================================================" << std::endl;

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

    typedef typename boost::mpl::transform<typename MssType::linear_esf,
                                           _impl::extract_functor>::type functors_list;

    struct arguments< functors_list, LoopIntervals, FunctorDoMethodsLookupMaps, range_sizes, LocalDomainList, Coords, ExecutionEngine > arguments_t;

    struct extra_arguments{
        typedef functors_list functors_list_t;
        typedef LoopIntervals loop_intervals_t;
        typedef FunctorDoMethodsLookupMaps functors_map_t
        typedef IntervalMap,
            typename LocalDomainType,
            typename Coords
            };


}
