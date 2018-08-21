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

#include <gridtools/c_bindings/generator.hpp>
#include <gridtools/common/gt_assert.hpp>

namespace gridtools {
    namespace c_bindings {
        namespace {
            class fortran_generics {
                std::map<char const *, std::vector<const char *>, _impl::c_string_less> m_procedures;

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
        } // namespace

        namespace _impl {
            void entities::add(char const *name, generator_t generator) {
                bool ok = m_generators.emplace(name, std::move(generator)).second;
                assert(ok);
            }

            std::ostream &operator<<(std::ostream &strm, entities const &obj) {
                for (auto &&item : obj.m_generators)
                    item.second(strm, item.first);
                return strm;
            }

            fortran_generic_registrar::fortran_generic_registrar(char const *generic_name, char const *concrete_name) {
                get_fortran_generics().add(generic_name, concrete_name);
            }

            template <>
            char const fortran_kind_name<bool>::value[] = "c_bool";
            template <>
            char const fortran_kind_name<int>::value[] = "c_int";
            template <>
            char const fortran_kind_name<short>::value[] = "c_short";
            template <>
            char const fortran_kind_name<long>::value[] = "c_long";
            template <>
            char const fortran_kind_name<long long>::value[] = "c_long_long";
            template <>
            char const fortran_kind_name<float>::value[] = "c_float";
            template <>
            char const fortran_kind_name<double>::value[] = "c_double";
            template <>
            char const fortran_kind_name<long double>::value[] = "c_long_double";
            template <>
            char const fortran_kind_name<signed char>::value[] = "c_signed_char";

            std::string fortran_array_element_type_name(gt_fortran_array_kind kind) {
                switch (kind) {
                case gt_fk_Bool:
                    return fortran_type_name<bool>();
                case gt_fk_Int:
                    return fortran_type_name<int>();
                case gt_fk_Short:
                    return fortran_type_name<short>();
                case gt_fk_Long:
                    return fortran_type_name<long>();
                case gt_fk_LongLong:
                    return fortran_type_name<long long>();
                case gt_fk_Float:
                    return fortran_type_name<float>();
                case gt_fk_Double:
                    return fortran_type_name<double>();
                case gt_fk_LongDouble:
                    return fortran_type_name<long double>();
                case gt_fk_SignedChar:
                    return fortran_type_name<signed char>();
                }
            }
        } // namespace _impl

        std::string wrap_line(const std::string &line, const std::string &prefix) {
            static constexpr uint_t max_line_length = 132;
            const std::string line_divider = " &";
            std::string ret = "";
            std::string current_prefix = prefix;

            auto it = line.begin();
            while (it + max_line_length - current_prefix.size() < line.end()) {
                auto next_it = it + max_line_length - line_divider.size() - current_prefix.size();
                while (*(next_it - 1) != ',') {
                    --next_it;
                    ASSERT_OR_THROW(next_it != line.begin() + 1, "Too long line cannot be wrapped");
                }

                ret.append(current_prefix);
                ret.append(it, next_it);
                ret.append(line_divider + "\n");

                it = next_it;
                // more indentation on next line
                current_prefix = prefix + "   ";
            }
            ret += current_prefix;
            ret.append(it, line.end());
            ret += '\n';
            return ret;
        }

        void generate_c_interface(std::ostream &strm) {
            strm << "\n#pragma once\n\n";
            strm << "#include <gridtools/c_bindings/array_descriptor.h>\n";
            strm << "#include <gridtools/c_bindings/handle.h>\n\n";
            strm << "#ifdef __cplusplus\n";
            strm << "extern \"C\" {\n";
            strm << "#endif\n\n";
            strm << _impl::get_entities<_impl::c_bindings_traits>();
            strm << "\n#ifdef __cplusplus\n";
            strm << "}\n";
            strm << "#endif\n";
        }

        void generate_fortran_interface(std::ostream &strm, std::string const &module_name) {
            strm << "\nmodule " << module_name << "\n";
            strm << "implicit none\n";
            strm << "  interface\n\n";
            strm << _impl::get_entities<_impl::fortran_bindings_traits>();
            strm << "\n  end interface\n";
            strm << get_fortran_generics();
            strm << "contains\n";
            strm << _impl::get_entities<_impl::fortran_wrapper_traits>();
            strm << "end\n";
        }
    } // namespace c_bindings
} // namespace gridtools
