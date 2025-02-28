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
# ifndef GT_PREPROCESSOR_REPETITION_ENUM_BINARY_PARAMS_HPP
# define GT_PREPROCESSOR_REPETITION_ENUM_BINARY_PARAMS_HPP
#
# include <gridtools/preprocessor/cat.hpp>
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/punctuation/comma_if.hpp>
# include <gridtools/preprocessor/repetition/repeat.hpp>
# include <gridtools/preprocessor/tuple/elem.hpp>
# include <gridtools/preprocessor/tuple/rem.hpp>
#
# /* GT_PP_ENUM_BINARY_PARAMS */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_ENUM_BINARY_PARAMS(count, p1, p2) GT_PP_REPEAT(count, GT_PP_ENUM_BINARY_PARAMS_M, (p1, p2))
# else
#    define GT_PP_ENUM_BINARY_PARAMS(count, p1, p2) GT_PP_ENUM_BINARY_PARAMS_I(count, p1, p2)
#    define GT_PP_ENUM_BINARY_PARAMS_I(count, p1, p2) GT_PP_REPEAT(count, GT_PP_ENUM_BINARY_PARAMS_M, (p1, p2))
# endif
#
# if GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_STRICT()
#    define GT_PP_ENUM_BINARY_PARAMS_M(z, n, pp) GT_PP_ENUM_BINARY_PARAMS_M_IM(z, n, GT_PP_TUPLE_REM_2 pp)
#    define GT_PP_ENUM_BINARY_PARAMS_M_IM(z, n, im) GT_PP_ENUM_BINARY_PARAMS_M_I(z, n, im)
# else
#    define GT_PP_ENUM_BINARY_PARAMS_M(z, n, pp) GT_PP_ENUM_BINARY_PARAMS_M_I(z, n, GT_PP_TUPLE_ELEM(2, 0, pp), GT_PP_TUPLE_ELEM(2, 1, pp))
# endif
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MSVC()
#    define GT_PP_ENUM_BINARY_PARAMS_M_I(z, n, p1, p2) GT_PP_ENUM_BINARY_PARAMS_M_II(z, n, p1, p2)
#    define GT_PP_ENUM_BINARY_PARAMS_M_II(z, n, p1, p2) GT_PP_COMMA_IF(n) p1 ## n p2 ## n
# else
#    define GT_PP_ENUM_BINARY_PARAMS_M_I(z, n, p1, p2) GT_PP_COMMA_IF(n) GT_PP_CAT(p1, n) GT_PP_CAT(p2, n)
# endif
#
# /* GT_PP_ENUM_BINARY_PARAMS_Z */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_ENUM_BINARY_PARAMS_Z(z, count, p1, p2) GT_PP_REPEAT_ ## z(count, GT_PP_ENUM_BINARY_PARAMS_M, (p1, p2))
# else
#    define GT_PP_ENUM_BINARY_PARAMS_Z(z, count, p1, p2) GT_PP_ENUM_BINARY_PARAMS_Z_I(z, count, p1, p2)
#    define GT_PP_ENUM_BINARY_PARAMS_Z_I(z, count, p1, p2) GT_PP_REPEAT_ ## z(count, GT_PP_ENUM_BINARY_PARAMS_M, (p1, p2))
# endif
#
# endif
