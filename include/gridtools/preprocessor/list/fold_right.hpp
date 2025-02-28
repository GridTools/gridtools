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
# /* Revised by Edward Diener (2020) */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_LIST_FOLD_RIGHT_HPP
# define GT_PREPROCESSOR_LIST_FOLD_RIGHT_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_STRICT()
#
# include <gridtools/preprocessor/cat.hpp>
# include <gridtools/preprocessor/control/while.hpp>
# include <gridtools/preprocessor/debug/error.hpp>
# include <gridtools/preprocessor/detail/auto_rec.hpp>
#
# if 0
#    define GT_PP_LIST_FOLD_RIGHT(op, state, list)
# endif
#
# define GT_PP_LIST_FOLD_RIGHT GT_PP_CAT(GT_PP_LIST_FOLD_RIGHT_, GT_PP_AUTO_REC(GT_PP_WHILE_P, 256))
#
# define GT_PP_LIST_FOLD_RIGHT_257(o, s, l) GT_PP_ERROR(0x0004)
#
# define GT_PP_LIST_FOLD_RIGHT_D(d, o, s, l) GT_PP_LIST_FOLD_RIGHT_ ## d(o, s, l)
# define GT_PP_LIST_FOLD_RIGHT_2ND GT_PP_LIST_FOLD_RIGHT
# define GT_PP_LIST_FOLD_RIGHT_2ND_D GT_PP_LIST_FOLD_RIGHT_D
#
# if GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    include <gridtools/preprocessor/list/detail/edg/fold_right.hpp>
# else
#    include <gridtools/preprocessor/list/detail/fold_right.hpp>
# endif
#
# else
#
# include <gridtools/preprocessor/arithmetic/dec.hpp>
# include <gridtools/preprocessor/cat.hpp>
# include <gridtools/preprocessor/control/while.hpp>
# include <gridtools/preprocessor/debug/error.hpp>
# include <gridtools/preprocessor/detail/auto_rec.hpp>
#
# if 0
#    define GT_PP_LIST_FOLD_RIGHT(op, state, list)
# endif
#
# include <gridtools/preprocessor/config/limits.hpp>
#
# if GT_PP_LIMIT_WHILE == 256
# define GT_PP_LIST_FOLD_RIGHT GT_PP_CAT(GT_PP_LIST_FOLD_RIGHT_, GT_PP_DEC(GT_PP_AUTO_REC(GT_PP_WHILE_P, 256)))
# define GT_PP_LIST_FOLD_RIGHT_257(o, s, l) GT_PP_ERROR(0x0004)
# elif GT_PP_LIMIT_WHILE == 512
# define GT_PP_LIST_FOLD_RIGHT GT_PP_CAT(GT_PP_LIST_FOLD_RIGHT_, GT_PP_DEC(GT_PP_AUTO_REC(GT_PP_WHILE_P, 512)))
# define GT_PP_LIST_FOLD_RIGHT_513(o, s, l) GT_PP_ERROR(0x0004)
# elif GT_PP_LIMIT_WHILE == 1024
# define GT_PP_LIST_FOLD_RIGHT GT_PP_CAT(GT_PP_LIST_FOLD_RIGHT_, GT_PP_DEC(GT_PP_AUTO_REC(GT_PP_WHILE_P, 1024)))
# define GT_PP_LIST_FOLD_RIGHT_1025(o, s, l) GT_PP_ERROR(0x0004)
# else
# error Incorrect value for the GT_PP_LIMIT_WHILE limit
# endif
#
# define GT_PP_LIST_FOLD_RIGHT_D(d, o, s, l) GT_PP_LIST_FOLD_RIGHT_ ## d(o, s, l)
# define GT_PP_LIST_FOLD_RIGHT_2ND GT_PP_LIST_FOLD_RIGHT
# define GT_PP_LIST_FOLD_RIGHT_2ND_D GT_PP_LIST_FOLD_RIGHT_D
#
# if GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    include <gridtools/preprocessor/list/detail/edg/fold_right.hpp>
# else
#    include <gridtools/preprocessor/list/detail/fold_right.hpp>
# endif
#
# endif
#
# endif
