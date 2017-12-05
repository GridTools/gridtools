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

#include <functional>
#include <ostream>
#include <string>

#include <boost/function_types/result_type.hpp>
#include <boost/function_types/parameter_types.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/type_index.hpp>

#include "function_wrapper.hpp"

namespace gridtools {
    namespace c_bindings {
        namespace _impl {

            template < class T >
            std::string get_type_name() {
                return boost::typeindex::type_id_with_cvr< T >().pretty_name();
            }

            template < class T >
            struct boxed {
                using type = boxed;
            };

            struct apply_to_param_f {
                template < class Fun, class T >
                void operator()(Fun &&fun, int &count, boxed< T >) const {
                    std::forward< Fun >(fun)(get_type_name< T >(), count);
                    ++count;
                }
            };

            template < class Signature, class Fun >
            void for_each_param(Fun &&fun) {
                namespace m = boost::mpl;
                int count = 0;
                m::for_each< typename boost::function_types::parameter_types< Signature >::type, boxed< m::_ > >(
                    std::bind(apply_to_param_f{}, std::forward< Fun >(fun), std::ref(count), std::placeholders::_1));
            };

            template < class Fun >
            std::ostream &write_delegation(std::ostream &strm, const std::string &name, const std::string &delegator) {
                namespace ft = boost::function_types;
                strm << get_type_name< typename ft::result_type< Fun >::type >() << " " << name << "(";
                for_each_param< Fun >([&](const std::string &type_name, int i) {
                    if (i)
                        strm << ", ";
                    strm << type_name << " arg_" << i;
                });
                strm << ") {\n";
                strm << "    return " << delegator << "(";
                for_each_param< Fun >([&](const std::string &type_name, int i) {
                    if (i)
                        strm << ", ";
                    strm << " arg_" << i;
                });
                strm << ");\n";
                strm << "}\n";
                return strm;
            }

            template < class Fun >
            std::ostream &write_declaration(std::ostream &strm, const std::string &name) {
                namespace ft = boost::function_types;
                strm << get_type_name< typename ft::result_type< Fun >::type >() << " " << name << "(";
                for_each_param< Fun >([&](const std::string &type_name, int i) {
                    if (i)
                        strm << ", ";
                    strm << type_name;
                });
                strm << ");\n";
                return strm;
            }
        }

        template < class Fun >
        std::ostream &write_definition(std::ostream &strm, const std::string &name, const std::string &impl) {
            return _impl::write_delegation< wrapped_t< Fun > >(
                strm, name, "::grigtools::c_bindings::wrap(" + impl + ")");
        }

        template < class Fun >
        std::ostream &write_declaration(std::ostream &strm, const std::string &name) {
            return _impl::write_declaration< wrapped_t< Fun > >(strm, name);
        }
    }
}
