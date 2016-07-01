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

#ifdef CXX11_ENABLED
#include "layout_map_cxx11.hpp"
#else
#include "layout_map_cxx03.hpp"
#endif

namespace gridtools {
    template <typename LM>
    struct reverse_map;

    template <short_t I1, short_t I2>
    struct reverse_map<layout_map<I1, I2> > {
        typedef layout_map<I2,I1> type;
    };

    template <short_t I1, short_t I2, short_t I3>
    struct reverse_map<layout_map<I1, I2,I3> > {
        template <short_t I, short_t Dummy>
        struct new_value;

        template <short_t Dummy>
        struct new_value<0, Dummy> {
            static const short_t value = 2;
        };

        template <short_t Dummy>
        struct new_value<1, Dummy> {
            static const short_t value = 1;
        };

        template <short_t Dummy>
        struct new_value<2, Dummy> {
            static const short_t value = 0;
        };

        typedef layout_map<new_value<I1,0>::value, new_value<I2,0>::value, new_value<I3,0>::value > type;
    };

    template <typename DATALO, typename PROCLO>
    struct layout_transform;

    template <short_t I1, short_t I2, short_t P1, short_t P2>
    struct layout_transform<layout_map<I1,I2>, layout_map<P1,P2> > {
        typedef layout_map<I1,I2> L1;
        typedef layout_map<P1,P2> L2;

        static const short_t N1 = boost::mpl::at_c<typename L1::layout_vector_t, P1>::type::value;
        static const short_t N2 = boost::mpl::at_c<typename L1::layout_vector_t, P2>::type::value;

        typedef layout_map<N1,N2> type;

    };

    template <short_t I1, short_t I2, short_t I3, short_t P1, short_t P2, short_t P3>
    struct layout_transform<layout_map<I1,I2,I3>, layout_map<P1,P2,P3> > {
        typedef layout_map<I1,I2,I3> L1;
        typedef layout_map<P1,P2,P3> L2;

        static const short_t N1 = boost::mpl::at_c<typename L1::layout_vector_t, P1>::type::value;
        static const short_t N2 = boost::mpl::at_c<typename L1::layout_vector_t, P2>::type::value;
        static const short_t N3 = boost::mpl::at_c<typename L1::layout_vector_t, P3>::type::value;

        typedef layout_map<N1,N2,N3> type;

    };

    template <short_t D>
    struct default_layout_map;

    template <>
    struct default_layout_map<1> {
        typedef layout_map<0> type;
    };

    template <>
    struct default_layout_map<2> {
        typedef layout_map<0,1> type;
    };

    template <>
    struct default_layout_map<3> {
        typedef layout_map<0,1,2> type;
    };

    template <>
    struct default_layout_map<4> {
        typedef layout_map<0,1,2,3> type;
    };
} // namespace gridtools
