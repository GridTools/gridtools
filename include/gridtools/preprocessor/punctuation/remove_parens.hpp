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
#ifndef GT_PREPROCESSOR_REMOVE_PARENS_HPP
#define GT_PREPROCESSOR_REMOVE_PARENS_HPP

#include <gridtools/preprocessor/config/config.hpp>
#include <gridtools/preprocessor/control/iif.hpp>
#include <gridtools/preprocessor/facilities/identity.hpp>
#include <gridtools/preprocessor/punctuation/is_begin_parens.hpp>
#include <gridtools/preprocessor/tuple/enum.hpp>

#define GT_PP_REMOVE_PARENS(param) \
    GT_PP_IIF \
      ( \
      GT_PP_IS_BEGIN_PARENS(param), \
      GT_PP_REMOVE_PARENS_DO, \
      GT_PP_IDENTITY \
      ) \
    (param)() \
/**/

#define GT_PP_REMOVE_PARENS_DO(param) \
  GT_PP_IDENTITY(GT_PP_TUPLE_ENUM(param)) \
/**/

#endif /* GT_PREPROCESSOR_REMOVE_PARENS_HPP */
