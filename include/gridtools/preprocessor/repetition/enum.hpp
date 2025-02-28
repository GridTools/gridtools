# /* Copyright (C) 2001
#  * Housemarque Oy
#  * http://www.housemarque.com
#  *
#  * Distributed under the Boost Software License, Version 1.0. (See
#  * accompanying file LICENSE_1_0.txt or copy at
#  * http://www.boost.org/LICENSE_1_0.txt)
#  */
#
# /* Revised by Paul Mensonides (2002) */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_REPETITION_ENUM_HPP
# define GT_PREPROCESSOR_REPETITION_ENUM_HPP
#
# include <gridtools/preprocessor/cat.hpp>
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/debug/error.hpp>
# include <gridtools/preprocessor/detail/auto_rec.hpp>
# include <gridtools/preprocessor/punctuation/comma_if.hpp>
# include <gridtools/preprocessor/repetition/repeat.hpp>
# include <gridtools/preprocessor/tuple/elem.hpp>
# include <gridtools/preprocessor/tuple/rem.hpp>
#
# /* GT_PP_ENUM */
#
# if 0
#    define GT_PP_ENUM(count, macro, data)
# endif
#
# define GT_PP_ENUM GT_PP_CAT(GT_PP_ENUM_, GT_PP_AUTO_REC(GT_PP_REPEAT_P, 4))
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_ENUM_1(c, m, d) GT_PP_REPEAT_1(c, GT_PP_ENUM_M_1, (m, d))
#    define GT_PP_ENUM_2(c, m, d) GT_PP_REPEAT_2(c, GT_PP_ENUM_M_2, (m, d))
#    define GT_PP_ENUM_3(c, m, d) GT_PP_REPEAT_3(c, GT_PP_ENUM_M_3, (m, d))
# else
#    define GT_PP_ENUM_1(c, m, d) GT_PP_ENUM_1_I(c, m, d)
#    define GT_PP_ENUM_2(c, m, d) GT_PP_ENUM_2_I(c, m, d)
#    define GT_PP_ENUM_3(c, m, d) GT_PP_ENUM_3_I(c, m, d)
#    define GT_PP_ENUM_1_I(c, m, d) GT_PP_REPEAT_1(c, GT_PP_ENUM_M_1, (m, d))
#    define GT_PP_ENUM_2_I(c, m, d) GT_PP_REPEAT_2(c, GT_PP_ENUM_M_2, (m, d))
#    define GT_PP_ENUM_3_I(c, m, d) GT_PP_REPEAT_3(c, GT_PP_ENUM_M_3, (m, d))
# endif
#
# define GT_PP_ENUM_4(c, m, d) GT_PP_ERROR(0x0003)
#
# if GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_STRICT()
#    define GT_PP_ENUM_M_1(z, n, md) GT_PP_ENUM_M_1_IM(z, n, GT_PP_TUPLE_REM_2 md)
#    define GT_PP_ENUM_M_2(z, n, md) GT_PP_ENUM_M_2_IM(z, n, GT_PP_TUPLE_REM_2 md)
#    define GT_PP_ENUM_M_3(z, n, md) GT_PP_ENUM_M_3_IM(z, n, GT_PP_TUPLE_REM_2 md)
#    define GT_PP_ENUM_M_1_IM(z, n, im) GT_PP_ENUM_M_1_I(z, n, im)
#    define GT_PP_ENUM_M_2_IM(z, n, im) GT_PP_ENUM_M_2_I(z, n, im)
#    define GT_PP_ENUM_M_3_IM(z, n, im) GT_PP_ENUM_M_3_I(z, n, im)
# else
#    define GT_PP_ENUM_M_1(z, n, md) GT_PP_ENUM_M_1_I(z, n, GT_PP_TUPLE_ELEM(2, 0, md), GT_PP_TUPLE_ELEM(2, 1, md))
#    define GT_PP_ENUM_M_2(z, n, md) GT_PP_ENUM_M_2_I(z, n, GT_PP_TUPLE_ELEM(2, 0, md), GT_PP_TUPLE_ELEM(2, 1, md))
#    define GT_PP_ENUM_M_3(z, n, md) GT_PP_ENUM_M_3_I(z, n, GT_PP_TUPLE_ELEM(2, 0, md), GT_PP_TUPLE_ELEM(2, 1, md))
# endif
#
# define GT_PP_ENUM_M_1_I(z, n, m, d) GT_PP_COMMA_IF(n) m(z, n, d)
# define GT_PP_ENUM_M_2_I(z, n, m, d) GT_PP_COMMA_IF(n) m(z, n, d)
# define GT_PP_ENUM_M_3_I(z, n, m, d) GT_PP_COMMA_IF(n) m(z, n, d)
#
# endif
