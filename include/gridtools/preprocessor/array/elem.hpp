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
# ifndef GT_PREPROCESSOR_ARRAY_ELEM_HPP
# define GT_PREPROCESSOR_ARRAY_ELEM_HPP
#
# include <gridtools/preprocessor/array/data.hpp>
# include <gridtools/preprocessor/array/size.hpp>
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/tuple/elem.hpp>
#
# /* GT_PP_ARRAY_ELEM */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_ARRAY_ELEM(i, array) GT_PP_TUPLE_ELEM(GT_PP_ARRAY_SIZE(array), i, GT_PP_ARRAY_DATA(array))
# else
#    define GT_PP_ARRAY_ELEM(i, array) GT_PP_ARRAY_ELEM_I(i, array)
#    define GT_PP_ARRAY_ELEM_I(i, array) GT_PP_TUPLE_ELEM(GT_PP_ARRAY_SIZE(array), i, GT_PP_ARRAY_DATA(array))
# endif
#
# endif
