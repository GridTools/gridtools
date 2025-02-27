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
#ifndef GT_PREPROCESSOR_DETAIL_IS_EMPTY_HPP
#define GT_PREPROCESSOR_DETAIL_IS_EMPTY_HPP

#include <gridtools/preprocessor/punctuation/is_begin_parens.hpp>

#if GT_PP_VARIADICS_MSVC

# pragma warning(once:4002)

#define GT_PP_DETAIL_IS_EMPTY_IIF_0(t, b) b
#define GT_PP_DETAIL_IS_EMPTY_IIF_1(t, b) t

#else

#define GT_PP_DETAIL_IS_EMPTY_IIF_0(t, ...) __VA_ARGS__
#define GT_PP_DETAIL_IS_EMPTY_IIF_1(t, ...) t

#endif

#if GT_PP_VARIADICS_MSVC && _MSC_VER <= 1400

#define GT_PP_DETAIL_IS_EMPTY_PROCESS(param) \
    GT_PP_IS_BEGIN_PARENS \
        ( \
        GT_PP_DETAIL_IS_EMPTY_NON_FUNCTION_C param () \
        ) \
/**/

#else

#define GT_PP_DETAIL_IS_EMPTY_PROCESS(...) \
    GT_PP_IS_BEGIN_PARENS \
        ( \
        GT_PP_DETAIL_IS_EMPTY_NON_FUNCTION_C __VA_ARGS__ () \
        ) \
/**/

#endif

#define GT_PP_DETAIL_IS_EMPTY_PRIMITIVE_CAT(a, b) a ## b
#define GT_PP_DETAIL_IS_EMPTY_IIF(bit) GT_PP_DETAIL_IS_EMPTY_PRIMITIVE_CAT(GT_PP_DETAIL_IS_EMPTY_IIF_,bit)
#define GT_PP_DETAIL_IS_EMPTY_NON_FUNCTION_C(...) ()

#endif /* GT_PREPROCESSOR_DETAIL_IS_EMPTY_HPP */
