# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Edward Diener 2014.                                    *
#  *     Distributed under the Boost Software License, Version 1.0. (See      *
#  *     accompanying file LICENSE_1_0.txt or copy at                         *
#  *     http://www.boost.org/LICENSE_1_0.txt)                                *
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_TUPLE_DETAIL_IS_SINGLE_RETURN_HPP
# define GT_PREPROCESSOR_TUPLE_DETAIL_IS_SINGLE_RETURN_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
#
# /* GT_PP_TUPLE_IS_SINGLE_RETURN */
#
# if GT_PP_VARIADICS_MSVC
# include <gridtools/preprocessor/control/iif.hpp>
# include <gridtools/preprocessor/facilities/is_1.hpp>
# include <gridtools/preprocessor/tuple/size.hpp>
# define GT_PP_TUPLE_IS_SINGLE_RETURN(sr,nsr,tuple)  \
    GT_PP_IIF(GT_PP_IS_1(GT_PP_TUPLE_SIZE(tuple)),sr,nsr) \
    /**/
# endif /* GT_PP_VARIADICS_MSVC */
#
# endif /* GT_PREPROCESSOR_TUPLE_DETAIL_IS_SINGLE_RETURN_HPP */
