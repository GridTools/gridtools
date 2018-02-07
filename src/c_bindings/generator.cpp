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

#include "c_bindings/generator.hpp"

#include <cassert>

namespace gridtools {
    namespace c_bindings {
        namespace {
            class fortran_generics {
                std::map< char const *, std::vector< const char * >, _impl::c_string_less > m_procedures;

              public:
                void add(char const *generic_name, char const *concrete_name) {
                    m_procedures[generic_name].push_back(concrete_name);
                }
                friend std::ostream &operator<<(std::ostream &strm, fortran_generics const &obj) {
                    for (auto &&item : obj.m_procedures) {
                        strm << "  interface " << item.first << "\n";
                        strm << "    procedure ";
                        bool need_comma = false;
                        for (auto &&procedure : item.second) {
                            if (need_comma)
                                strm << ", ";
                            strm << procedure;
                            need_comma = true;
                        }
                        strm << "\n";
                        strm << "  end interface\n";
                    }
                    return strm;
                }
            };

            fortran_generics &get_fortran_generics() {
                static fortran_generics obj;
                return obj;
            }
        }

        namespace _impl {
            void declarations::add(char const *name, generator_t generator) {
                bool ok = m_generators.emplace(name, std::move(generator)).second;
                assert(ok);
            }

            std::ostream &operator<<(std::ostream &strm, declarations const &obj) {
                for (auto &&item : obj.m_generators)
                    item.second(strm, item.first);
                return strm;
            }

            fortran_generic_registrar::fortran_generic_registrar(char const *generic_name, char const *concrete_name) {
                get_fortran_generics().add(generic_name, concrete_name);
            }

            template <>
            char const fortran_kind_name< bool >::value[] = "c_bool";
            template <>
            char const fortran_kind_name< int >::value[] = "c_int";
            template <>
            char const fortran_kind_name< short >::value[] = "c_short";
            template <>
            char const fortran_kind_name< long >::value[] = "c_long";
            template <>
            char const fortran_kind_name< long long >::value[] = "c_long_long";
            template <>
            char const fortran_kind_name< float >::value[] = "c_float";
            template <>
            char const fortran_kind_name< double >::value[] = "c_double";
            template <>
            char const fortran_kind_name< long double >::value[] = "c_long_double";
            template <>
            char const fortran_kind_name< signed char >::value[] = "c_signed_char";
        }

        void generate_c_interface(std::ostream &strm) {
            strm << "\n#pragma once\n\n";
            strm << "#include <c_bindings/handle.h>\n\n";
            strm << "#ifdef __cplusplus\n";
            strm << "extern \"C\" {\n";
            strm << "#endif\n\n";
            strm << _impl::get_declarations< _impl::c_traits >();
            strm << "\n#ifdef __cplusplus\n";
            strm << "}\n";
            strm << "#endif\n";
        }

        void generate_fortran_interface(std::ostream &strm, std::string const &module_name) {
            strm << "\nmodule " << module_name << "\n";
            strm << "implicit none\n";
            strm << "  interface\n\n";
            strm << _impl::get_declarations< _impl::fortran_traits >();
            strm << "\n  end interface\n";
            strm << get_fortran_generics();
            strm << "end\n";
        }
    }
}
