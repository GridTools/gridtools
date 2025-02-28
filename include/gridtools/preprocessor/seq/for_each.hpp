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
# ifndef GT_PREPROCESSOR_SEQ_FOR_EACH_HPP
# define GT_PREPROCESSOR_SEQ_FOR_EACH_HPP
#
# include <gridtools/preprocessor/arithmetic/dec.hpp>
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/control/if.hpp>
# include <gridtools/preprocessor/control/iif.hpp>
# include <gridtools/preprocessor/repetition/for.hpp>
# include <gridtools/preprocessor/seq/seq.hpp>
# include <gridtools/preprocessor/seq/size.hpp>
# include <gridtools/preprocessor/seq/detail/is_empty.hpp>
# include <gridtools/preprocessor/tuple/elem.hpp>
# include <gridtools/preprocessor/tuple/rem.hpp>
#
# /* GT_PP_SEQ_FOR_EACH */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_SEQ_FOR_EACH(macro, data, seq) GT_PP_SEQ_FOR_EACH_DETAIL_CHECK(macro, data, seq)
# else
#    define GT_PP_SEQ_FOR_EACH(macro, data, seq) GT_PP_SEQ_FOR_EACH_D(macro, data, seq)
#    define GT_PP_SEQ_FOR_EACH_D(macro, data, seq) GT_PP_SEQ_FOR_EACH_DETAIL_CHECK(macro, data, seq)
# endif
#
#    define GT_PP_SEQ_FOR_EACH_DETAIL_CHECK_EXEC(macro, data, seq) GT_PP_FOR((macro, data, seq, GT_PP_SEQ_SIZE(seq)), GT_PP_SEQ_FOR_EACH_P, GT_PP_SEQ_FOR_EACH_O, GT_PP_SEQ_FOR_EACH_M)
#    define GT_PP_SEQ_FOR_EACH_DETAIL_CHECK_EMPTY(macro, data, seq)
#
#    define GT_PP_SEQ_FOR_EACH_DETAIL_CHECK(macro, data, seq) \
        GT_PP_IIF \
            ( \
            GT_PP_SEQ_DETAIL_IS_NOT_EMPTY(seq), \
            GT_PP_SEQ_FOR_EACH_DETAIL_CHECK_EXEC, \
            GT_PP_SEQ_FOR_EACH_DETAIL_CHECK_EMPTY \
            ) \
        (macro, data, seq) \
/**/
#
# define GT_PP_SEQ_FOR_EACH_P(r, x) GT_PP_TUPLE_ELEM(4, 3, x)
#
# if GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_STRICT()
#    define GT_PP_SEQ_FOR_EACH_O(r, x) GT_PP_SEQ_FOR_EACH_O_I x
# else
#    define GT_PP_SEQ_FOR_EACH_O(r, x) GT_PP_SEQ_FOR_EACH_O_I(GT_PP_TUPLE_ELEM(4, 0, x), GT_PP_TUPLE_ELEM(4, 1, x), GT_PP_TUPLE_ELEM(4, 2, x), GT_PP_TUPLE_ELEM(4, 3, x))
# endif
#
# define GT_PP_SEQ_FOR_EACH_O_I(macro, data, seq, sz) \
    GT_PP_SEQ_FOR_EACH_O_I_DEC(macro, data, seq, GT_PP_DEC(sz)) \
/**/
# define GT_PP_SEQ_FOR_EACH_O_I_DEC(macro, data, seq, sz) \
    ( \
    macro, \
    data, \
    GT_PP_IF \
        ( \
        sz, \
        GT_PP_SEQ_FOR_EACH_O_I_TAIL, \
        GT_PP_SEQ_FOR_EACH_O_I_NIL \
        ) \
    (seq), \
    sz \
    ) \
/**/
# define GT_PP_SEQ_FOR_EACH_O_I_TAIL(seq) GT_PP_SEQ_TAIL(seq)
# define GT_PP_SEQ_FOR_EACH_O_I_NIL(seq) GT_PP_NIL
#
# if GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_STRICT()
#    define GT_PP_SEQ_FOR_EACH_M(r, x) GT_PP_SEQ_FOR_EACH_M_IM(r, GT_PP_TUPLE_REM_4 x)
#    define GT_PP_SEQ_FOR_EACH_M_IM(r, im) GT_PP_SEQ_FOR_EACH_M_I(r, im)
# else
#    define GT_PP_SEQ_FOR_EACH_M(r, x) GT_PP_SEQ_FOR_EACH_M_I(r, GT_PP_TUPLE_ELEM(4, 0, x), GT_PP_TUPLE_ELEM(4, 1, x), GT_PP_TUPLE_ELEM(4, 2, x), GT_PP_TUPLE_ELEM(4, 3, x))
# endif
#
# define GT_PP_SEQ_FOR_EACH_M_I(r, macro, data, seq, sz) macro(r, data, GT_PP_SEQ_HEAD(seq))
#
# /* GT_PP_SEQ_FOR_EACH_R */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_SEQ_FOR_EACH_R(r, macro, data, seq) GT_PP_SEQ_FOR_EACH_DETAIL_CHECK_R(r, macro, data, seq)
# else
#    define GT_PP_SEQ_FOR_EACH_R(r, macro, data, seq) GT_PP_SEQ_FOR_EACH_R_I(r, macro, data, seq)
#    define GT_PP_SEQ_FOR_EACH_R_I(r, macro, data, seq) GT_PP_SEQ_FOR_EACH_DETAIL_CHECK_R(r, macro, data, seq)
# endif
#
#    define GT_PP_SEQ_FOR_EACH_DETAIL_CHECK_EXEC_R(r, macro, data, seq) GT_PP_FOR_ ## r((macro, data, seq, GT_PP_SEQ_SIZE(seq)), GT_PP_SEQ_FOR_EACH_P, GT_PP_SEQ_FOR_EACH_O, GT_PP_SEQ_FOR_EACH_M)
#    define GT_PP_SEQ_FOR_EACH_DETAIL_CHECK_EMPTY_R(r, macro, data, seq)
#
#    define GT_PP_SEQ_FOR_EACH_DETAIL_CHECK_R(r, macro, data, seq) \
        GT_PP_IIF \
            ( \
            GT_PP_SEQ_DETAIL_IS_NOT_EMPTY(seq), \
            GT_PP_SEQ_FOR_EACH_DETAIL_CHECK_EXEC_R, \
            GT_PP_SEQ_FOR_EACH_DETAIL_CHECK_EMPTY_R \
            ) \
        (r, macro, data, seq) \
/**/
#
# endif
