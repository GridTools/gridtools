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
#include <boost/mpl/quote.hpp>
#include <boost/mpl/transform.hpp>

namespace gridtools {

    /*
     * @struct is_sequence_of
     * metafunction that determines if a mpl sequence is a sequence of types determined by the filter
     * @param TSeq sequence to query
     * @param TPred filter that determines the condition
     */
    template < typename Seq, template < typename > class Lambda >
    struct apply_to_sequence {
        typedef boost::mpl::quote1< Lambda > lambda_t;

        typedef typename boost::mpl::transform< Seq, lambda_t >::type type;
    };

} // namespace gridtools
