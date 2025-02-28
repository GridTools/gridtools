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
# ifndef GT_PREPROCESSOR_SEQ_SEQ_HPP
# define GT_PREPROCESSOR_SEQ_SEQ_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/seq/elem.hpp>
#
# /* GT_PP_SEQ_HEAD */
#
# define GT_PP_SEQ_HEAD(seq) GT_PP_SEQ_ELEM(0, seq)
#
# /* GT_PP_SEQ_TAIL */
#
# if GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC()
#    define GT_PP_SEQ_TAIL(seq) GT_PP_SEQ_TAIL_1((seq))
#    define GT_PP_SEQ_TAIL_1(par) GT_PP_SEQ_TAIL_2 ## par
#    define GT_PP_SEQ_TAIL_2(seq) GT_PP_SEQ_TAIL_I ## seq
# elif GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MSVC()
#    define GT_PP_SEQ_TAIL(seq) GT_PP_SEQ_TAIL_ID(GT_PP_SEQ_TAIL_I seq)
#    define GT_PP_SEQ_TAIL_ID(id) id
# elif GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_SEQ_TAIL(seq) GT_PP_SEQ_TAIL_D(seq)
#    define GT_PP_SEQ_TAIL_D(seq) GT_PP_SEQ_TAIL_I seq
# else
#    define GT_PP_SEQ_TAIL(seq) GT_PP_SEQ_TAIL_I seq
# endif
#
# define GT_PP_SEQ_TAIL_I(x)
#
# /* GT_PP_SEQ_NIL */
#
# define GT_PP_SEQ_NIL(x) (x)
#
# endif
