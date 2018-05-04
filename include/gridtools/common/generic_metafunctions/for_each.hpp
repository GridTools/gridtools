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

#include "../host_device.hpp"

namespace gridtools {
    namespace _impl {
        template < class List >
        struct for_each_f;

        template < template < class... > class L, class T, class... Ts >
        struct for_each_f< L< T, Ts... > > {
            template < class Fun >
            GT_FUNCTION void operator()(Fun const &fun) const {
                (void)(int[]){((void)fun(T{}), 0), ((void)fun(Ts{}), 0)...};
            }
        };

        // Specialization for empty loops as nvcc refuses to compile the normal version in device code
        template < template < class... > class L >
        struct for_each_f< L<> > {
            template < class Fun >
            GT_FUNCTION void operator()(Fun const &) const {}
        };

        template < class List >
        struct host_for_each_f;

        template < template < class... > class L, class... Ts >
        struct host_for_each_f< L< Ts... > > {
            template < class Fun >
            void operator()(Fun const &fun) const {
                (void)(int[]){((void)fun(Ts{}), 0)...};
            }
        };
    }

    /** \ingroup common
        @{
        \ingroup allmeta
        @{
        \defgroup foreach For Each
        @{
    */
    /// Calls fun(T{}) for each element of the type list List.
    template < class List, class Fun >
    GT_FUNCTION Fun for_each(Fun const &fun) {
        _impl::for_each_f< List >{}(fun);
        return fun;
    };

    // TODO(anstaf): avoid copying the same thing with and without GT_FUNCTION.
    //               Possible solution is boost preprocessor vertical repetition pattern
    //               it should be for_each, host::for_each and device::for_each
    //               for CUDA they will be different, for others all are aliases to host::for_each
    //               The same pattern could be applied for all template functions that we use both in
    //               device and host context.
    template < class List, class Fun >
    Fun host_for_each(Fun const &fun) {
        _impl::host_for_each_f< List >{}(fun);
        return fun;
    };

    /** @} */
    /** @} */
    /** @} */
}
