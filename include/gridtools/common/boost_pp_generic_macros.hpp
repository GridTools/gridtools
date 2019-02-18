/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \addtogroup common
    @{
*/

/** \addtogroup fixes
    @{
*/

#if (!defined(BOOST_PP_VARIADICS) || (BOOST_PP_VARIADICS < 1))
// defining BOOST_PP_VARIADICS 1 here might be too late, therefore we leave it to the user
#error \
    "GRIDTOOLS ERROR=> For the repository you need to \"#define BOOST_PP_VARIADICS 1\" before the first include of any boost preprocessor file.")
#endif

#include <boost/preprocessor/seq.hpp>
#include <boost/preprocessor/tuple.hpp>
#include <boost/variant.hpp>

/**
 * @def GT_PP_TUPLE_ELEM_FROM_SEQ_AS_ENUM(tuple_elem_id, seq_of_tuples)
 * @brief Returns a comma separated list of the i-th tuple elements form a list of tuples.
 * Example: GT_PP_TUPLE_ELEM_FROM_SEQ_AS_ENUM( 1, ((double,3))((int,5)) ) -> 3,5
 *
 * @param tuple_elem_id id of the tuple elements
 * @param seq_of_tuples sequence of tuples
 */

/** @cond */
#define GT_PP_TUPLE_ELEM_FROM_SEQ_AS_ENUM_helper(r, tuple_elem_id, n, tuple) \
    BOOST_PP_COMMA_IF(n) BOOST_PP_TUPLE_ELEM(tuple_elem_id, tuple)
/** @endcond */
#define GT_PP_TUPLE_ELEM_FROM_SEQ_AS_ENUM(tuple_elem_id, seq_of_tuples) \
    BOOST_PP_SEQ_FOR_EACH_I(GT_PP_TUPLE_ELEM_FROM_SEQ_AS_ENUM_helper, tuple_elem_id, seq_of_tuples)

/**
 * @def GT_PP_MAKE_VARIANT(variant_name, types)
 * @brief Generates a boost::variant with user-defined conversion to each of its types
 * @param variant_name name of the class to be generated
 * @param types BOOST_PP sequence (can be a sequence of tuples where the first tuple element is the type name),
 * e.g. ((int))(double))
 * or ((int),(3))((double),(1))
 */

/** @cond */
#define GT_PP_MAKE_VARIANT_conversion(r, data, tuple) \
    operator BOOST_PP_TUPLE_ELEM(0, tuple)() const { return boost::get<BOOST_PP_TUPLE_ELEM(0, tuple)>(*this); }

#define GT_PP_MAKE_VARIANT_conversions_loop(types) BOOST_PP_SEQ_FOR_EACH(GT_PP_MAKE_VARIANT_conversion, ~, types)
/** @endcond */

#define GT_PP_MAKE_VARIANT(variant_name, types)                                               \
    class variant_name : public boost::variant<GT_PP_TUPLE_ELEM_FROM_SEQ_AS_ENUM(0, types)> { \
      public:                                                                                 \
        using boost::variant<GT_PP_TUPLE_ELEM_FROM_SEQ_AS_ENUM(0, types)>::variant;           \
        GT_PP_MAKE_VARIANT_conversions_loop(types)                                            \
    };

/**
 * @def GT_PP_EMPTY(...)
 * @brief Takes any number of parameters and expands to nothing.
 */
#define GT_PP_EMPTY(...)

/** @} */
/** @} */
