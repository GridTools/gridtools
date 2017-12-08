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
#include <map>
#include <ostream>
#include <string>

#include <boost/function_types/result_type.hpp>
#include <boost/function_types/parameter_types.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/type_index.hpp>

namespace gridtools {
    namespace c_bindings {

        enum class language { c, fortran };

        namespace _impl {

            class declarations {
                using generator_t = std::function< void(std::ostream &, std::string const &) >;
                std::map< std::string, generator_t > m_generators;

              public:
                void add(std::string name, generator_t generator);
                friend std::ostream &operator<<(std::ostream &strm, declarations const &);
            };

            template < class >
            declarations &get_declarations() {
                static declarations obj;
                return obj;
            }

            template < class T, class = void >
            struct recursive_remove_cv : std::remove_cv< T > {};

            template < class T >
            struct recursive_remove_cv< T, typename std::enable_if< std::is_pointer< T >::value >::type > {
                using type = typename recursive_remove_cv< typename std::remove_pointer< T >::type >::type *;
            };

            template < class T >
            std::string get_c_type_name() {
                return boost::typeindex::type_id< typename recursive_remove_cv< T >::type >().pretty_name();
            }

            template < class T >
            struct boxed {
                using type = boxed;
            };

            struct apply_to_param_f {
                template < class Fun, class T >
                void operator()(Fun &&fun, int &count, boxed< T >) const {
                    std::forward< Fun >(fun)(get_c_type_name< T >(), count);
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
            std::ostream &write_declaration(std::ostream &strm, const std::string &name) {
                namespace ft = boost::function_types;
                strm << get_c_type_name< typename ft::result_type< Fun >::type >() << " " << name << "(";
                for_each_param< Fun >([&](const std::string &type_name, int i) {
                    if (i)
                        strm << ", ";
                    strm << type_name;
                });
                return strm << ");\n";
            }

            struct c_traits {
                template < class Fun >
                static void generate_declaration(std::ostream &strm, std::string const &name) {
                    write_declaration< Fun >(strm, name);
                }
                static const char m_prologue[];
                static const char m_epilogue[];
            };

            struct fortran_traits {
                template < class Fun >
                static void generate_declaration(std::ostream &strm, std::string const &name) {}

                static const char m_prologue[];
                static const char m_epilogue[];
            };

            template < class Traits, class Fun >
            void add_declaration(std::string const &name) {
                get_declarations< Traits >().add(name, Traits::template generate_declaration< Fun >);
            }

            template < class Fun >
            struct registrar {
                registrar(const std::string &name) {
                    add_declaration< _impl::c_traits, Fun >(name);
                    add_declaration< _impl::fortran_traits, Fun >(name);
                }
            };

            template < class Traits >
            void generate_interface(std::ostream &strm) {
                strm << Traits::m_prologue << _impl::get_declarations< Traits >() << Traits::m_epilogue;
            }
        }

        /// Outputs the content of the C compatible header with the declarations added by GT_ADD_GENERATED_DECLARATION
        template < class Strm >
        Strm generate_c_interface(Strm &&strm) {
            _impl::generate_interface< _impl::c_traits >(strm);
            return std::forward< Strm >(strm);
        }

        /// Outputs the content of the Fortran module with the declarations added by GT_ADD_GENERATED_DECLARATION
        template < class Strm >
        Strm generate_fortran_interface(Strm &&strm) {
            _impl::generate_interface< _impl::fortran_traits >(strm);
            return std::forward< Strm >(strm);
        }
    }
}

/**
 *  Registers the function that for declaration generations.
 *  Users should not this directly.
 */
#define GT_ADD_GENERATED_DECLARATION(signature, name) \
    static ::gridtools::c_bindings::_impl::registrar< signature > generated_declaration_registrar_##name(#name)
