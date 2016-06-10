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

namespace gridtools {

    namespace _impl {

        // check in a sequence of AMss that if there is reduction, it is placed at the end
        template < typename AMssSeq >
        struct check_mss_seq {
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of< AMssSeq, is_amss_descriptor >::value), "Error");
            typedef
                typename boost::mpl::find_if< AMssSeq, is_reduction_descriptor< boost::mpl::_ > >::type check_iter_t;

            GRIDTOOLS_STATIC_ASSERT((boost::is_same< check_iter_t, typename boost::mpl::end< AMssSeq >::type >::value ||
                                        check_iter_t::pos::value == boost::mpl::size< AMssSeq >::value - 1),
                "Error deducing the reduction. Check that if there is a reduction, this appears in the last mss");
            typedef notype type;
        };

        /**
         * helper struct to deduce the type of a reduction and extract the initial value of a reduction passed via API.
         * specialization returns a notype when argument passed is not a reduction
         */
        template < typename... Mss >
        struct reduction_helper;

        template < typename First, typename... Mss >
        struct reduction_helper< First, Mss... > : reduction_helper< Mss... > {
            typedef typename reduction_helper< Mss... >::reduction_type_t reduction_type_t;
            GRIDTOOLS_STATIC_ASSERT((is_mss_descriptor< First >::value),
                "Only Mss are allowed in the make_computations,"
                "except for reduction in the last position");
            static reduction_type_t extract_initial_value(First, Mss... args) {
                return reduction_helper< Mss... >::extract_initial_value(args...);
            }
        };

        template < typename T1, typename T2, typename Tag >
        struct reduction_helper< condition< T1, T2, Tag > > {

            typedef condition< T1, T2, Tag > cond_t;

            typedef notype reduction_type_t;
            static notype extract_initial_value(cond_t) { return 0; }
        };

        template < typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence >
        struct reduction_helper< mss_descriptor< ExecutionEngine, EsfDescrSequence, CacheSequence > > {
            typedef mss_descriptor< ExecutionEngine, EsfDescrSequence, CacheSequence > mss_t;
            typedef notype reduction_type_t;
            static notype extract_initial_value(mss_t) { return 0; }
        };

        template < typename ExecutionEngine, typename BinOp, typename EsfDescrSequence >
        struct reduction_helper< reduction_descriptor< ExecutionEngine, BinOp, EsfDescrSequence > > {
            typedef reduction_descriptor< ExecutionEngine, BinOp, EsfDescrSequence > mss_t;
            typedef typename mss_t::reduction_type_t reduction_type_t;

            static typename mss_t::reduction_type_t extract_initial_value(mss_t &red) { return red.get(); }
        };

    } // namespace _impl
} // namespace gridtools
