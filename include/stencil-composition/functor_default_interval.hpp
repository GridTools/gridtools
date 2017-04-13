/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

namespace gridtools {
    /**@brief decorates the user function with a defaiult interval, in case no interval was specified by the user

       A SFINAE mechanism detects wether the user gave the vertical interval as input to the Do method,
       and when this is not the case it wraps the functor inside this decoratpr, passing the whole axis to it.
       The first and last points are removed from the axis (the GridTools API works with exclusive intervals). So
       the functor_decorator spans the whole domain embedded in the vertical axis.
       \tparam F the user functor
       \tparam Axis the vertical axis
     */
    template < typename Functor, typename Axis >
    struct functor_default_interval {
        GRIDTOOLS_STATIC_ASSERT(is_interval< Axis >::value, "Internal error");
        static const constexpr int_t to_offset = Axis::ToLevel::Offset::value;
        static const constexpr uint_t to_splitter = Axis::ToLevel::Splitter::value;
        static const constexpr int_t from_offset = Axis::FromLevel::Offset::value;
        static const constexpr uint_t from_splitter = Axis::FromLevel::Splitter::value;

        // NOTE: because of API convention the default interval representing the whole vertical axis must have the same
        // splitters as Axis,
        // and offsets which are strictly enclosed in those defined by the Axis template argument
        // NOTE: the offsets cannot be 0 because of API conventions,
        // for this reason whe have to devise a special case for when the "from" offset in the vertical Axis is -1
        // (in that case the interval representing the whole axis must start from 1 instead of 0, so we have to add
        // -1+2=1)
        typedef gridtools::interval< level< from_splitter, from_offset >,
            level< to_splitter, (to_offset != 1) ? to_offset - 1 : to_offset - 2 > > default_interval;

        typedef typename Functor::arg_list arg_list;

        template < typename Eval >
        GT_FUNCTION static void Do(Eval const &eval_, default_interval) {
            Functor::Do(eval_);
        }
    };
    template < typename T >
    struct is_functor_default_interval : boost::mpl::false_ {};

    template < typename T, typename A >
    struct is_functor_default_interval< functor_default_interval< T, A > > : boost::mpl::true_ {};
} // namespace gridtools
