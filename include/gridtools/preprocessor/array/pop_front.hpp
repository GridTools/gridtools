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
# ifndef GT_PREPROCESSOR_ARRAY_POP_FRONT_HPP
# define GT_PREPROCESSOR_ARRAY_POP_FRONT_HPP
#
# include <gridtools/preprocessor/arithmetic/dec.hpp>
# include <gridtools/preprocessor/arithmetic/inc.hpp>
# include <gridtools/preprocessor/array/elem.hpp>
# include <gridtools/preprocessor/array/size.hpp>
# include <gridtools/preprocessor/repetition/enum.hpp>
# include <gridtools/preprocessor/repetition/deduce_z.hpp>
#
# /* GT_PP_ARRAY_POP_FRONT */
#
# define GT_PP_ARRAY_POP_FRONT(array) GT_PP_ARRAY_POP_FRONT_Z(GT_PP_DEDUCE_Z(), array)
#
# /* GT_PP_ARRAY_POP_FRONT_Z */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_ARRAY_POP_FRONT_Z(z, array) GT_PP_ARRAY_POP_FRONT_I(z, GT_PP_ARRAY_SIZE(array), array)
# else
#    define GT_PP_ARRAY_POP_FRONT_Z(z, array) GT_PP_ARRAY_POP_FRONT_Z_D(z, array)
#    define GT_PP_ARRAY_POP_FRONT_Z_D(z, array) GT_PP_ARRAY_POP_FRONT_I(z, GT_PP_ARRAY_SIZE(array), array)
# endif
#
# define GT_PP_ARRAY_POP_FRONT_I(z, size, array) (GT_PP_DEC(size), (GT_PP_ENUM_ ## z(GT_PP_DEC(size), GT_PP_ARRAY_POP_FRONT_M, array)))
# define GT_PP_ARRAY_POP_FRONT_M(z, n, data) GT_PP_ARRAY_ELEM(GT_PP_INC(n), data)
#
# endif
