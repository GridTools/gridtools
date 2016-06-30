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
#ifndef _TRANSLATE_H_
#define _TRANSLATE_H_

#include "../../common/layout_map.hpp"

namespace gridtools {

    template < int D, typename map = typename default_layout_map< 3 >::type >
    struct translate_t;

    //   template <>
    //   struct translate_t<2, layout_map<0,1,2> > {
    //     typedef layout_map<0,1,2> map_type;
    //     inline int operator()(int I, int J) {return (I+1)*3+J+1;}
    //   };

    template <>
    struct translate_t< 2, layout_map< 0, 1 > > {
        typedef layout_map< 0, 1 > map_type;
        inline int operator()(int I, int J) { return (I + 1) * 3 + J + 1; }
    };

    template <>
    struct translate_t< 2, layout_map< 1, 0 > > {
        typedef layout_map< 1, 0 > map_type;
        inline int operator()(int I, int J) { return (J + 1) * 3 + I + 1; }
    };

    template <>
    struct translate_t< 3, layout_map< 0, 1, 2 > > {
        typedef layout_map< 0, 1, 2 > map_type;
        inline int operator()(int I, int J, int K) { return (K + 1) * 9 + (J + 1) * 3 + I + 1; }
    };

    template <>
    struct translate_t< 3, layout_map< 2, 1, 0 > > {
        typedef layout_map< 2, 1, 0 > map_type;
        inline int operator()(int I, int J, int K) { return (I + 1) * 9 + (J + 1) * 3 + K + 1; }
    };

    template <>
    struct translate_t< 3, layout_map< 1, 2, 0 > > {
        typedef layout_map< 1, 2, 0 > map_type;
        inline int operator()(int I, int J, int K) { return (J + 1) * 9 + (I + 1) * 3 + K + 1; }
    };

    template <>
    struct translate_t< 3, layout_map< 0, 2, 1 > > {
        typedef layout_map< 0, 2, 1 > map_type;
        inline int operator()(int I, int J, int K) { return (K + 1) * 9 + (I + 1) * 3 + J + 1; }
    };

    template <>
    struct translate_t< 3, layout_map< 2, 0, 1 > > {
        typedef layout_map< 2, 0, 1 > map_type;
        inline int operator()(int I, int J, int K) { return (I + 1) * 9 + (K + 1) * 3 + J + 1; }
    };

    template <>
    struct translate_t< 3, layout_map< 1, 0, 2 > > {
        typedef layout_map< 1, 0, 2 > map_type;
        inline int operator()(int I, int J, int K) { return (J + 1) * 9 + (K + 1) * 3 + I + 1; }
    };
}

#endif
