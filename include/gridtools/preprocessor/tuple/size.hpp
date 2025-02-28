# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Edward Diener 2011.                                    *
#  *     (C) Copyright Paul Mensonides 2011.                                  *
#  *     Distributed under the Boost Software License, Version 1.0. (See      *
#  *     accompanying file LICENSE_1_0.txt or copy at                         *
#  *     http://www.boost.org/LICENSE_1_0.txt)                                *
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_TUPLE_SIZE_HPP
# define GT_PREPROCESSOR_TUPLE_SIZE_HPP
#
# include <gridtools/preprocessor/cat.hpp>
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/control/if.hpp>
# include <gridtools/preprocessor/variadic/has_opt.hpp>
# include <gridtools/preprocessor/variadic/size.hpp>
#
# if GT_PP_VARIADIC_HAS_OPT()
#     if GT_PP_VARIADICS_MSVC
#         define GT_PP_TUPLE_SIZE(tuple) GT_PP_TUPLE_SIZE_CHECK(GT_PP_CAT(GT_PP_VARIADIC_SIZE tuple,))
#     else
#         define GT_PP_TUPLE_SIZE(tuple) GT_PP_TUPLE_SIZE_CHECK(GT_PP_VARIADIC_SIZE tuple)
#     endif
#     define GT_PP_TUPLE_SIZE_CHECK(size) GT_PP_IF(size,size,1)
# elif GT_PP_VARIADICS_MSVC
#     define GT_PP_TUPLE_SIZE(tuple) GT_PP_CAT(GT_PP_VARIADIC_SIZE tuple,)
# else
#     define GT_PP_TUPLE_SIZE(tuple) GT_PP_VARIADIC_SIZE tuple
# endif
#
# endif
