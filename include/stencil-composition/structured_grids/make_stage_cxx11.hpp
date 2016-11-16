/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include "common/generic_metafunctions/variadic_to_vector.hpp"
#include "stencil-composition/mss_metafunctions.hpp"

namespace gridtools {

    /*!
       \fn esf_descriptor<ESF, ...> make_esf(plc1, plc2, plc3, ...)
       \brief Function to create a Elementary Stencil Function
       \param plc{i} placeholder which represents the i-th argument to the functor ESF

       Use this function to associate a stencil functor (stage) to
       arguments (actually, placeholders to arguments)
     */

    /*!
       \fn mss_descriptor<...> make_esf(ExecutionEngine, esf1, esf2, ...)
       \brief Function to create a Multistage Stencil that can then be executed
       \param esf{i}  i-th Elementary Stencil Function created with make_esf or a list specified as independent ESFs
       created with make independent

       Use this function to create a multi-stage stencil computation
     */

    /*!
       \fn independent_esf<...> make_independent(esf1, esf2, ...)
       \brief Function to create a list of independent Elementary Styencil Functions

       \param esf{i}  (must be i>=2) The max{i} Elementary Stencil Functions in the argument list will be treated as
       independent

       Function to create a list of independent Elementary Styencil Functions. This is used to let the library compute
       tight bounds on blocks to be used by backends
     */

    template < typename ESF, typename... ExtraArgs >
    esf_descriptor< ESF, boost::mpl::vector< ExtraArgs... > > make_stage(ExtraArgs &&... /*args_*/) {
        GRIDTOOLS_STATIC_ASSERT((accumulate(logical_and(), is_arg< ExtraArgs >::value...)), "Malformed make_esf");
#ifdef PEDANTIC // find a way to enable this check also with generic accessors
        GRIDTOOLS_STATIC_ASSERT((sizeof...(ExtraArgs) == boost::mpl::size<typename ESF::arg_list>::value), "wrong number of arguments passed to the make_esf");
#endif
        return esf_descriptor< ESF, boost::mpl::vector< ExtraArgs... > >();
    }

    template < typename ESF, typename Staggering, typename... ExtraArgs >
    esf_descriptor< ESF, boost::mpl::vector< ExtraArgs... >, Staggering > make_stage(ExtraArgs &&... args_) {
        GRIDTOOLS_STATIC_ASSERT((accumulate(logical_and(), is_arg< ExtraArgs >::value...)), "Malformed make_esf");
        return esf_descriptor< ESF, boost::mpl::vector< ExtraArgs... >, Staggering >();
    }

} // namespace gridtools
