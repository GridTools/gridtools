# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Paul Mensonides 2002.
#  *     Distributed under the Boost Software License, Version 1.0. (See
#  *     accompanying file LICENSE_1_0.txt or copy at
#  *     http://www.boost.org/LICENSE_1_0.txt)
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_SEQ_TRANSFORM_HPP
# define GT_PREPROCESSOR_SEQ_TRANSFORM_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/seq/fold_left.hpp>
# include <gridtools/preprocessor/seq/seq.hpp>
# include <gridtools/preprocessor/tuple/elem.hpp>
# include <gridtools/preprocessor/tuple/rem.hpp>
#
# /* GT_PP_SEQ_TRANSFORM */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_SEQ_TRANSFORM(op, data, seq) GT_PP_SEQ_TAIL(GT_PP_TUPLE_ELEM(3, 2, GT_PP_SEQ_FOLD_LEFT(GT_PP_SEQ_TRANSFORM_O, (op, data, (nil)), seq)))
# else
#    define GT_PP_SEQ_TRANSFORM(op, data, seq) GT_PP_SEQ_TRANSFORM_I(op, data, seq)
#    define GT_PP_SEQ_TRANSFORM_I(op, data, seq) GT_PP_SEQ_TAIL(GT_PP_TUPLE_ELEM(3, 2, GT_PP_SEQ_FOLD_LEFT(GT_PP_SEQ_TRANSFORM_O, (op, data, (nil)), seq)))
# endif
#
# if GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_STRICT()
#    define GT_PP_SEQ_TRANSFORM_O(s, state, elem) GT_PP_SEQ_TRANSFORM_O_IM(s, GT_PP_TUPLE_REM_3 state, elem)
#    define GT_PP_SEQ_TRANSFORM_O_IM(s, im, elem) GT_PP_SEQ_TRANSFORM_O_I(s, im, elem)
# else
#    define GT_PP_SEQ_TRANSFORM_O(s, state, elem) GT_PP_SEQ_TRANSFORM_O_I(s, GT_PP_TUPLE_ELEM(3, 0, state), GT_PP_TUPLE_ELEM(3, 1, state), GT_PP_TUPLE_ELEM(3, 2, state), elem)
# endif
#
# define GT_PP_SEQ_TRANSFORM_O_I(s, op, data, res, elem) (op, data, res (op(s, data, elem)))
#
# /* GT_PP_SEQ_TRANSFORM_S */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_SEQ_TRANSFORM_S(s, op, data, seq) GT_PP_SEQ_TAIL(GT_PP_TUPLE_ELEM(3, 2, GT_PP_SEQ_FOLD_LEFT_ ## s(GT_PP_SEQ_TRANSFORM_O, (op, data, (nil)), seq)))
# else
#    define GT_PP_SEQ_TRANSFORM_S(s, op, data, seq) GT_PP_SEQ_TRANSFORM_S_I(s, op, data, seq)
#    define GT_PP_SEQ_TRANSFORM_S_I(s, op, data, seq) GT_PP_SEQ_TAIL(GT_PP_TUPLE_ELEM(3, 2, GT_PP_SEQ_FOLD_LEFT_ ## s(GT_PP_SEQ_TRANSFORM_O, (op, data, (nil)), seq)))
# endif
#
# endif
