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

#include <type_traits>

#include <stdbool.h>

#include <boost/any.hpp>

extern "C" {

struct gt_handle {
    boost::any m_value;
};

void gt_release(gt_handle const *obj) { delete obj; }
}

namespace gridtools {
    namespace c_bindings {
        namespace _impl {

            template < class T, class = void >
            struct result_converted_to_c;

            template < class T >
            struct result_converted_to_c< T,
                typename std::enable_if< std::is_void< T >::value || std::is_arithmetic< T >::value >::type > {
                using type = T;
            };

            template < class T >
            struct result_converted_to_c< T,
                typename std::enable_if< std::is_class< typename std::remove_reference< T >::type >::value >::type > {
                using type = gt_handle *;
            };

            template < class T, class = void >
            struct param_converted_to_c;

            template < class T >
            struct param_converted_to_c< T, typename std::enable_if< std::is_arithmetic< T >::value >::type > {
                using type = T;
            };

            template < class T >
            struct param_converted_to_c< T *, typename std::enable_if< std::is_arithmetic< T >::value >::type > {
                using type = T *;
            };

            template < class T >
            struct param_converted_to_c< T &, typename std::enable_if< std::is_arithmetic< T >::value >::type > {
                using type = T *;
            };

            template < class T >
            struct param_converted_to_c< T,
                typename std::enable_if< std::is_class<
                    typename std::remove_pointer< typename std::remove_reference< T >::type >::type >::value >::type > {
                using type = gt_handle *;
            };

            template < class T, typename std::enable_if< std::is_arithmetic< T >::value, int >::type = 0 >
            T convert_to_c(T obj) {
                return obj;
            }

            template < class T,
                typename std::enable_if< std::is_class< typename std::remove_reference< T >::type >::value,
                    int >::type = 0 >
            gt_handle *convert_to_c(T &&obj) {
                return new gt_handle{std::forward< T >(obj)};
            }

            template < class T >
            using result_converted_to_c_t = typename result_converted_to_c< T >::type;
            template < class T >
            using param_converted_to_c_t = typename param_converted_to_c< T >::type;

            template < class T,
                typename std::enable_if< std::is_integral< typename std::remove_pointer< T >::type >::value,
                    int >::type = 0 >
            T convert_from_c(T obj) {
                return obj;
            };

            template < class T,
                typename std::enable_if< std::is_reference< T >::value &&
                                             std::is_integral< typename std::remove_reference< T >::type >::value,
                    int >::type = 0 >
            T convert_from_c(typename std::remove_reference< T >::type *obj) {
                return *obj;
            };

            template < class T >
            T convert_from_c(gt_handle *obj) {
                return boost::any_cast< T >(obj->m_value);
            }

            template < class T >
            struct wrapped_f;

            template < class R, class... Params >
            struct wrapped_f< R (*)(Params...) > {
                R (*m_fun)(Params...);
                result_converted_to_c_t< R > operator()(param_converted_to_c_t< Params >... args) const {
                    return convert_to_c(m_fun(convert_from_c< Params >(args)...));
                }
            };

            template < class... Params >
            struct wrapped_f< void (*)(Params...) > {
                void (*m_fun)(Params...);
                void operator()(param_converted_to_c_t< Params >... args) const {
                    return m_fun(convert_from_c< Params >(args)...);
                }
            };

            template < class T >
            struct wrapped;

            template < class R, class... Params >
            struct wrapped< R (*)(Params...) > {
                using type = typename wrapped< R(Params...) >::type;
            };

            template < class R, class... Params >
            struct wrapped< R(Params...) > {
                using type = result_converted_to_c_t< R >(param_converted_to_c_t< Params >...);
            };
        }

        template < class T >
        constexpr _impl::wrapped_f< T > wrap(T obj) {
            return {obj};
        }

        template < class T >
        using wrapped_t = typename _impl::wrapped< T >::type;
    }
}
