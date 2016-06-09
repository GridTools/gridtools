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
#include <map>
#include <string>
#include <iostream>

#include <cstdio>
#include <cassert>

#include <GCL.hpp>

/*
namespace gridtools {

// initialize static instance_ to NULL
template<> stats_collector<1>* stats_collector<1>::instance_ = 0;
template<> stats_collector<2>* stats_collector<2>::instance_ = 0;
template<> stats_collector<3>* stats_collector<3>::instance_ = 0;

// convenient handles for the singleton instances for 2D and 3D grids
stats_collector<3> &stats_collector_3D = *stats_collector<3>::instance();
stats_collector<2> &stats_collector_2D = *stats_collector<2>::instance();
} // namespace gridtools
*/
