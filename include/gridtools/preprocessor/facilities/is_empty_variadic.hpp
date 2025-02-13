# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Edward Diener 2014,2019.
#  *     Distributed under the Boost Software License, Version 1.0. (See
#  *     accompanying file LICENSE_1_0.txt or copy at
#  *     http://www.boost.org/LICENSE_1_0.txt)
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_FACILITIES_IS_EMPTY_VARIADIC_HPP
# define GT_PREPROCESSOR_FACILITIES_IS_EMPTY_VARIADIC_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/punctuation/is_begin_parens.hpp>
# include <gridtools/preprocessor/facilities/detail/is_empty.hpp>
#
#if GT_PP_VARIADICS_MSVC && _MSC_VER <= 1400
#
#define GT_PP_IS_EMPTY(param) \
    GT_PP_DETAIL_IS_EMPTY_IIF \
      ( \
      GT_PP_IS_BEGIN_PARENS \
        ( \
        param \
        ) \
      ) \
      ( \
      GT_PP_IS_EMPTY_ZERO, \
      GT_PP_DETAIL_IS_EMPTY_PROCESS \
      ) \
    (param) \
/**/
#define GT_PP_IS_EMPTY_ZERO(param) 0
# else
# if defined(__cplusplus) && __cplusplus > 201703L
# include <gridtools/preprocessor/variadic/has_opt.hpp>
#define GT_PP_IS_EMPTY(...) \
    GT_PP_DETAIL_IS_EMPTY_IIF \
      ( \
      GT_PP_VARIADIC_HAS_OPT() \
      ) \
      ( \
      GT_PP_IS_EMPTY_OPT, \
      GT_PP_IS_EMPTY_NO_OPT \
      ) \
    (__VA_ARGS__) \
/**/
#define GT_PP_IS_EMPTY_FUNCTION2(...) \
    __VA_OPT__(0,) 1 \
/**/
#define GT_PP_IS_EMPTY_FUNCTION(...) \
    GT_PP_IS_EMPTY_FUNCTION2(__VA_ARGS__) \
/**/
#define GT_PP_IS_EMPTY_OPT(...) \
    GT_PP_VARIADIC_HAS_OPT_ELEM0(GT_PP_IS_EMPTY_FUNCTION(__VA_ARGS__),) \
/**/
# else
#define GT_PP_IS_EMPTY(...) \
    GT_PP_IS_EMPTY_NO_OPT(__VA_ARGS__) \
/**/
# endif /* defined(__cplusplus) && __cplusplus > 201703L */
#define GT_PP_IS_EMPTY_NO_OPT(...) \
    GT_PP_DETAIL_IS_EMPTY_IIF \
      ( \
      GT_PP_IS_BEGIN_PARENS \
        ( \
        __VA_ARGS__ \
        ) \
      ) \
      ( \
      GT_PP_IS_EMPTY_ZERO, \
      GT_PP_DETAIL_IS_EMPTY_PROCESS \
      ) \
    (__VA_ARGS__) \
/**/
#define GT_PP_IS_EMPTY_ZERO(...) 0
# endif /* GT_PP_VARIADICS_MSVC && _MSC_VER <= 1400 */
# endif /* GT_PREPROCESSOR_FACILITIES_IS_EMPTY_VARIADIC_HPP */
