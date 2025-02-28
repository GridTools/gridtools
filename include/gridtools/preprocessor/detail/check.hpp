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
# ifndef GT_PREPROCESSOR_DETAIL_CHECK_HPP
# define GT_PREPROCESSOR_DETAIL_CHECK_HPP
#
# include <gridtools/preprocessor/cat.hpp>
# include <gridtools/preprocessor/config/config.hpp>
#
# /* GT_PP_CHECK */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC()
#    define GT_PP_CHECK(x, type) GT_PP_CHECK_D(x, type)
# else
#    define GT_PP_CHECK(x, type) GT_PP_CHECK_OO((x, type))
#    define GT_PP_CHECK_OO(par) GT_PP_CHECK_D ## par
# endif
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MSVC() && ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_DMC()
#    define GT_PP_CHECK_D(x, type) GT_PP_CHECK_1(GT_PP_CAT(GT_PP_CHECK_RESULT_, type x))
#    define GT_PP_CHECK_1(chk) GT_PP_CHECK_2(chk)
#    define GT_PP_CHECK_2(res, _) res
# elif GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MSVC()
#    define GT_PP_CHECK_D(x, type) GT_PP_CHECK_1(type x)
#    define GT_PP_CHECK_1(chk) GT_PP_CHECK_2(chk)
#    define GT_PP_CHECK_2(chk) GT_PP_CHECK_3((GT_PP_CHECK_RESULT_ ## chk))
#    define GT_PP_CHECK_3(im) GT_PP_CHECK_5(GT_PP_CHECK_4 im)
#    define GT_PP_CHECK_4(res, _) res
#    define GT_PP_CHECK_5(res) res
# else /* DMC */
#    define GT_PP_CHECK_D(x, type) GT_PP_CHECK_OO((type x))
#    define GT_PP_CHECK_OO(par) GT_PP_CHECK_0 ## par
#    define GT_PP_CHECK_0(chk) GT_PP_CHECK_1(GT_PP_CAT(GT_PP_CHECK_RESULT_, chk))
#    define GT_PP_CHECK_1(chk) GT_PP_CHECK_2(chk)
#    define GT_PP_CHECK_2(res, _) res
# endif
#
# define GT_PP_CHECK_RESULT_1 1, GT_PP_NIL
#
# endif
