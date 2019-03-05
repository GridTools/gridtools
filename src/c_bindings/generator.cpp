/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
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
                    const std::string prefix = "    ";
                    for (auto &&item : obj.m_procedures) {
                        strm << "  interface " << item.first << "\n";
                        std::string line = "";
                        line += "procedure ";
                        bool need_comma = false;
                        for (auto &&procedure : item.second) {
                            if (need_comma)
                                line += ", ";
                            line += procedure;
                            need_comma = true;
                        }
                        strm << wrap_line(line, prefix);
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
                default:
                    assert(false && "Invalid element kind");
                    return {};
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
                    GT_ASSERT_OR_THROW(next_it != line.begin() + 1, "Too long line cannot be wrapped");
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
            strm << "// This file is generated!\n";
            strm << "#pragma once\n\n";
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
            strm << "! This file is generated!\n";
            strm << "module " << module_name << "\n";
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
