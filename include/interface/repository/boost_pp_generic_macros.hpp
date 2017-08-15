/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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

#if (!defined(BOOST_PP_VARIADICS) || (BOOST_PP_VARIADICS < 1))
// defining BOOST_PP_VARIADICS 1 here might be too late, therefore we leave it to the user
#error \
    "GRIDTOOLS ERROR=> For the repository you need to \"#define BOOST_PP_VARIADICS 1\" before the first include of any boost preprocessor file.")
#endif

#include <unordered_map>
#include "boost/variant.hpp"
#include "boost/preprocessor/seq.hpp"
#include <boost/preprocessor/tuple.hpp>

/*
 * @brief GRIDTOOLS_PP_SEQ_DOUBLE_PARENS adds extra parenthesis to a make a valid BOOST_PP sequence
 * e.g. (int, 2)(double, 3) -> ((int, 2)) ((double, 3))
 */
#define GRIDTOOLS_PP_SEQ_DOUBLE_PARENS_internal_0(...) ((__VA_ARGS__)) GRIDTOOLS_PP_SEQ_DOUBLE_PARENS_internal_1
#define GRIDTOOLS_PP_SEQ_DOUBLE_PARENS_internal_1(...) ((__VA_ARGS__)) GRIDTOOLS_PP_SEQ_DOUBLE_PARENS_internal_0
#define GRIDTOOLS_PP_SEQ_DOUBLE_PARENS_internal_0_END
#define GRIDTOOLS_PP_SEQ_DOUBLE_PARENS_internal_1_END
#define GRIDTOOLS_PP_SEQ_DOUBLE_PARENS(seq) BOOST_PP_CAT(GRIDTOOLS_PP_SEQ_DOUBLE_PARENS_internal_0 seq, _END)

/**
 * @brief GRIDTOOLS_PP_TUPLE_ELEM_FROM_SEQ_AS_ENUM returns a comma separated list of the i-th tuple elements form a list
 * of tuples
 * @param tuple_elem_id id of the tuple elements
 * @param seq_of_tuples sequence of tuples
 * @example GRIDTOOLS_PP_TUPLE_ELEM_FROM_SEQ_AS_ENUM( 1, ((double,3))((int,5)) ) -> 3,5
 */
#define GRIDTOOLS_PP_TUPLE_ELEM_FROM_SEQ_AS_ENUM_helper(r, tuple_elem_id, n, tuple) \
    BOOST_PP_COMMA_IF(n) BOOST_PP_TUPLE_ELEM(tuple_elem_id, tuple)

#define GRIDTOOLS_PP_TUPLE_ELEM_FROM_SEQ_AS_ENUM(tuple_elem_id, seq_of_tuples) \
    BOOST_PP_SEQ_FOR_EACH_I(GRIDTOOLS_PP_TUPLE_ELEM_FROM_SEQ_AS_ENUM_helper, tuple_elem_id, seq_of_tuples)

/*
 * @brief GRIDTOOLS_PP_MAKE_VARIANT generates a boost::variant with user-defined conversion to each of its types
 * @param variant_name name of the class to be generated
 * @param types BOOST_PP sequence (can be a sequence of tuples where the first tuple element is the type name),
 * e.g. ((int))(double))
 * or ((int),(3))((double),(1))
 */
#define GRIDTOOLS_PP_MAKE_VARIANT_conversion(r, data, tuple) \
    operator BOOST_PP_TUPLE_ELEM(0, tuple)() const { return boost::get< BOOST_PP_TUPLE_ELEM(0, tuple) >(*this); }

#define GRIDTOOLS_PP_MAKE_VARIANT_conversions_loop(GTREPO_FIELDTYPES_SEQ) \
    BOOST_PP_SEQ_FOR_EACH(GRIDTOOLS_PP_MAKE_VARIANT_conversion, ~, GTREPO_FIELDTYPES_SEQ)

#define GRIDTOOLS_PP_MAKE_VARIANT(variant_name, types)                                                 \
    class variant_name : public boost::variant< GRIDTOOLS_PP_TUPLE_ELEM_FROM_SEQ_AS_ENUM(0, types) > { \
      public:                                                                                          \
        using boost::variant< GRIDTOOLS_PP_TUPLE_ELEM_FROM_SEQ_AS_ENUM(0, types) >::variant;           \
        GRIDTOOLS_PP_MAKE_VARIANT_conversions_loop(types)                                              \
    };

/*
 * @brief takes any number of parameters and expands to nothing
 */
#define GRIDTOOLS_PP_EMPTY(...)
