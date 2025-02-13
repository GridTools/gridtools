# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Edward Diener 2014.
#  *     Distributed under the Boost Software License, Version 1.0. (See
#  *     accompanying file LICENSE_1_0.txt or copy at
#  *     http://www.boost.org/LICENSE_1_0.txt)
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
#ifndef GT_PREPROCESSOR_DETAIL_IS_BEGIN_PARENS_HPP
#define GT_PREPROCESSOR_DETAIL_IS_BEGIN_PARENS_HPP

#if GT_PP_VARIADICS_MSVC

#include <gridtools/preprocessor/facilities/empty.hpp>

#define GT_PP_DETAIL_VD_IBP_CAT(a, b) GT_PP_DETAIL_VD_IBP_CAT_I(a, b)
#define GT_PP_DETAIL_VD_IBP_CAT_I(a, b) GT_PP_DETAIL_VD_IBP_CAT_II(a ## b)
#define GT_PP_DETAIL_VD_IBP_CAT_II(res) res

#define GT_PP_DETAIL_IBP_SPLIT(i, ...) \
    GT_PP_DETAIL_VD_IBP_CAT(GT_PP_DETAIL_IBP_PRIMITIVE_CAT(GT_PP_DETAIL_IBP_SPLIT_,i)(__VA_ARGS__),GT_PP_EMPTY()) \
/**/

#define GT_PP_DETAIL_IBP_IS_VARIADIC_C(...) 1 1

#else

#define GT_PP_DETAIL_IBP_SPLIT(i, ...) \
    GT_PP_DETAIL_IBP_PRIMITIVE_CAT(GT_PP_DETAIL_IBP_SPLIT_,i)(__VA_ARGS__) \
/**/

#define GT_PP_DETAIL_IBP_IS_VARIADIC_C(...) 1

#endif /* GT_PP_VARIADICS_MSVC */

#define GT_PP_DETAIL_IBP_SPLIT_0(a, ...) a
#define GT_PP_DETAIL_IBP_SPLIT_1(a, ...) __VA_ARGS__

#define GT_PP_DETAIL_IBP_CAT(a, ...) GT_PP_DETAIL_IBP_PRIMITIVE_CAT(a,__VA_ARGS__)
#define GT_PP_DETAIL_IBP_PRIMITIVE_CAT(a, ...) a ## __VA_ARGS__

#define GT_PP_DETAIL_IBP_IS_VARIADIC_R_1 1,
#define GT_PP_DETAIL_IBP_IS_VARIADIC_R_GT_PP_DETAIL_IBP_IS_VARIADIC_C 0,

#endif /* GT_PREPROCESSOR_DETAIL_IS_BEGIN_PARENS_HPP */
