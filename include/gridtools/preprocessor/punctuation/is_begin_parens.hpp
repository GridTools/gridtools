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
# ifndef GT_PREPROCESSOR_IS_BEGIN_PARENS_HPP
# define GT_PREPROCESSOR_IS_BEGIN_PARENS_HPP

#include <gridtools/preprocessor/config/config.hpp>
#include <gridtools/preprocessor/punctuation/detail/is_begin_parens.hpp>

#if GT_PP_VARIADICS_MSVC && _MSC_VER <= 1400

#define GT_PP_IS_BEGIN_PARENS(param) \
    GT_PP_DETAIL_IBP_SPLIT \
      ( \
      0, \
      GT_PP_DETAIL_IBP_CAT \
        ( \
        GT_PP_DETAIL_IBP_IS_VARIADIC_R_, \
        GT_PP_DETAIL_IBP_IS_VARIADIC_C param \
        ) \
      ) \
/**/

#else

#define GT_PP_IS_BEGIN_PARENS(...) \
    GT_PP_DETAIL_IBP_SPLIT \
      ( \
      0, \
      GT_PP_DETAIL_IBP_CAT \
        ( \
        GT_PP_DETAIL_IBP_IS_VARIADIC_R_, \
        GT_PP_DETAIL_IBP_IS_VARIADIC_C __VA_ARGS__ \
        ) \
      ) \
/**/

#endif /* GT_PP_VARIADICS_MSVC && _MSC_VER <= 1400 */
#endif /* GT_PREPROCESSOR_IS_BEGIN_PARENS_HPP */
