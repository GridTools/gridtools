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
# ifndef GT_PREPROCESSOR_DEBUG_ERROR_HPP
# define GT_PREPROCESSOR_DEBUG_ERROR_HPP
#
# include <gridtools/preprocessor/cat.hpp>
# include <gridtools/preprocessor/config/config.hpp>
#
# /* GT_PP_ERROR */
#
# if GT_PP_CONFIG_ERRORS
#    define GT_PP_ERROR(code) GT_PP_CAT(GT_PP_ERROR_, code)
# endif
#
# define GT_PP_ERROR_0x0000 GT_PP_ERROR(0x0000, GT_PP_INDEX_OUT_OF_BOUNDS)
# define GT_PP_ERROR_0x0001 GT_PP_ERROR(0x0001, GT_PP_WHILE_OVERFLOW)
# define GT_PP_ERROR_0x0002 GT_PP_ERROR(0x0002, GT_PP_FOR_OVERFLOW)
# define GT_PP_ERROR_0x0003 GT_PP_ERROR(0x0003, GT_PP_REPEAT_OVERFLOW)
# define GT_PP_ERROR_0x0004 GT_PP_ERROR(0x0004, GT_PP_LIST_FOLD_OVERFLOW)
# define GT_PP_ERROR_0x0005 GT_PP_ERROR(0x0005, GT_PP_SEQ_FOLD_OVERFLOW)
# define GT_PP_ERROR_0x0006 GT_PP_ERROR(0x0006, GT_PP_ARITHMETIC_OVERFLOW)
# define GT_PP_ERROR_0x0007 GT_PP_ERROR(0x0007, GT_PP_DIVISION_BY_ZERO)
#
# endif
