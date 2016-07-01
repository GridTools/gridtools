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
#pragma once

/**
   @file traits classes for the various grid topologies
*/

namespace gridtools {
    namespace topology {

        /**@brief cartesian topology

           \tparam the local layout map, i.e. defining the order of the dimensions
         */
        template < typename Layout >
        struct cartesian {};
    } // namespace topology

    template<typename TopologyType>
    class cell_topology{};

    template < typename Layout >
    class cell_topology< topology::cartesian< Layout > > {
    public:
        static const ushort_t space_dimensions=Layout::length;
    };
} // namespace gridtools
