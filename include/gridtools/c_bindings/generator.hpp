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

#include <cassert>
#include <cstring>
#include <functional>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <boost/function_types/parameter_types.hpp>
#include <boost/function_types/result_type.hpp>
#include <boost/optional.hpp>
#include <boost/type_index.hpp>

#include "../common/generic_metafunctions/copy_into_variadic.hpp"
#include "../common/generic_metafunctions/is_there_in_sequence_if.hpp"
#include "../meta/transform.hpp"

#include "function_wrapper.hpp"

namespace gridtools {
    namespace c_bindings {

        std::string wrap_line(const std::string &line, const std::string &prefix);

        namespace _impl {

            struct c_string_less {
                bool operator()(char const *lhs, char const *rhs) const { return strcmp(lhs, rhs) < 0; }
            };

            class entities {
                using generator_t = std::function<void(std::ostream &, char const *)>;
                std::map<char const *, generator_t, c_string_less> m_generators;

              public:
                void add(char const *name, generator_t generator);
                friend std::ostream &operator<<(std::ostream &strm, entities const &obj);
            };

            template <class>
            entities &get_entities() {
                static entities obj;
                return obj;
            }

            template <class T, class = void>
            struct recursive_remove_cv : std::remove_cv<T> {};

            template <class T>
            struct recursive_remove_cv<T, typename std::enable_if<std::is_pointer<T>::value>::type> {
                using type = typename recursive_remove_cv<typename std::remove_pointer<T>::type>::type *;
            };

            struct get_c_type_name_f {
                template <class T>
                std::string operator()() const {
                    return boost::typeindex::type_id<typename recursive_remove_cv<T>::type>().pretty_name();
                }
            };

            template <class T>
            std::string get_c_type_name() {
                return boost::typeindex::type_id<typename recursive_remove_cv<T>::type>().pretty_name();
            }

            template <class TypeToStr, class Fun>
            struct for_each_param_helper_f {
                TypeToStr m_type_to_str;
                Fun m_fun;
                int &m_count;

                template <class T>
                void operator()() const {
                    m_fun(m_type_to_str.template operator()<T>(), m_count);
                    ++m_count;
                }
            };

            template <class Signature,
                class TypeToStr,
                class Fun,
                class Params =
                    copy_into_variadic<typename boost::function_types::parameter_types<Signature>::type, std::tuple<>>>
            void for_each_param(TypeToStr &&type_to_str, Fun &&fun) {
                int count = 0;
                for_each_type<Params>(for_each_param_helper_f<TypeToStr, Fun>{
                    std::forward<TypeToStr>(type_to_str), std::forward<Fun>(fun), count});
            };

            template <class CSignature>
            std::ostream &write_c_binding(std::ostream &strm, char const *name) {
                namespace ft = boost::function_types;
                strm << get_c_type_name<typename ft::result_type<CSignature>::type>() << " " << name << "(";
                for_each_param<CSignature>(get_c_type_name_f{}, [&](const std::string &type_name, int i) {
                    if (i)
                        strm << ", ";
                    strm << type_name;
                });
                return strm << ");\n";
            }

            template <class>
            struct fortran_kind_name {
                static char const value[];
            };

            template <>
            char const fortran_kind_name<bool>::value[];
            template <>
            char const fortran_kind_name<int>::value[];
            template <>
            char const fortran_kind_name<short>::value[];
            template <>
            char const fortran_kind_name<long>::value[];
            template <>
            char const fortran_kind_name<long long>::value[];
            template <>
            char const fortran_kind_name<float>::value[];
            template <>
            char const fortran_kind_name<double>::value[];
            template <>
            char const fortran_kind_name<long double>::value[];
            template <>
            char const fortran_kind_name<signed char>::value[];

            template <class T,
                typename std::enable_if<std::is_same<typename std::decay<T>::type, bool>::value, int>::type = 0>
            std::string fortran_type_name() {
                using decayed_t = typename std::decay<T>::type;
                return std::string("logical(") + fortran_kind_name<decayed_t>::value + ")";
            }
            template <class T,
                typename std::enable_if<std::is_integral<T>::value &&
                                            !std::is_same<typename std::decay<T>::type, bool>::value,
                    int>::type = 0>
            std::string fortran_type_name() {
                using signed_decayed_t = typename std::make_signed<typename std::decay<T>::type>::type;
                return std::string("integer(") + fortran_kind_name<signed_decayed_t>::value + ")";
            }

            template <class T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
            std::string fortran_type_name() {
                using decayed_t = typename std::decay<T>::type;
                return std::string("real(") + fortran_kind_name<decayed_t>::value + ")";
            }

            template <class T, typename std::enable_if<std::is_pointer<T>::value, int>::type = 0>
            std::string fortran_type_name() {
                return "type(c_ptr)";
            }

            template <class T,
                typename std::enable_if<!std::is_pointer<T>::value && !std::is_integral<T>::value &&
                                            !std::is_floating_point<T>::value,
                    int>::type = 0>
            std::string fortran_type_name() {
                assert("Unsupported fortran type." && false);
                return "";
            }

            template <class T, typename std::enable_if<std::is_void<T>::value, int>::type = 0>
            std::string fortran_function_specifier() {
                return "subroutine";
            }

            template <class T, typename std::enable_if<!std::is_void<T>::value, int>::type = 0>
            std::string fortran_function_specifier() {
                return "function";
            }

            template <class T, typename std::enable_if<std::is_void<T>::value, int>::type = 0>
            std::string fortran_return_type() {
                return fortran_function_specifier<T>();
            }

            template <class T, typename std::enable_if<!std::is_void<T>::value, int>::type = 0>
            std::string fortran_return_type() {
                return fortran_type_name<T>() + " " + fortran_function_specifier<T>();
            }

            std::string fortran_array_element_type_name(gt_fortran_array_kind kind);

            struct ignore_type_f {
                template <class T>
                std::string operator()() const {
                    return "";
                }
            };

            struct fortran_param_type_from_c_f {

                template <class CType,
                    typename std::enable_if<std::is_same<CType, gt_fortran_array_descriptor *>::value, int>::type = 0>
                std::string operator()() const {
                    return "type(gt_fortran_array_descriptor)";
                }

                template <class CType,
                    typename std::enable_if<!std::is_same<CType, gt_fortran_array_descriptor *>::value &&
                                                (!std::is_pointer<CType>::value ||
                                                    std::is_class<typename std::remove_pointer<CType>::type>::value),
                        int>::type = 0>
                std::string operator()() const {
                    return fortran_type_name<CType>() + ", value";
                }

                template <class CType,
                    typename std::enable_if<std::is_pointer<CType>::value &&
                                                std::is_arithmetic<typename std::remove_pointer<CType>::type>::value,
                        int>::type = 0>
                std::string operator()() const {
                    return fortran_type_name<typename std::remove_pointer<CType>::type>() + ", dimension(*)";
                }
                template <class CType,
                    typename std::enable_if<std::is_pointer<CType>::value &&
                                                !std::is_arithmetic<typename std::remove_pointer<CType>::type>::value &&
                                                !std::is_class<typename std::remove_pointer<CType>::type>::value,
                        int>::type = 0>
                std::string operator()() const {
                    return "type(c_ptr)";
                }
            };
            struct fortran_param_type_from_cpp_f {

                template <class CppType,
                    class CType = param_converted_to_c_t<CppType>,
                    typename std::enable_if<std::is_same<CType, gt_fortran_array_descriptor *>::value &&
                                                is_fortran_array_wrappable<CppType>::value,
                        int>::type = 0>
                std::string operator()() const {
                    static const gt_fortran_array_descriptor meta =
                        get_fortran_view_meta((add_pointer_t<CppType>){nullptr});
                    std::string dimensions = "dimension(";
                    for (int i = 0; i < meta.rank; ++i) {
                        if (i)
                            dimensions += ",";
                        dimensions += ":";
                    }
                    dimensions += ")";
                    return fortran_array_element_type_name(meta.type) + ", " + dimensions;
                }

                template <class CppType,
                    class CType = param_converted_to_c_t<CppType>,
                    typename std::enable_if<!std::is_same<CType, gt_fortran_array_descriptor *>::value ||
                                                !is_fortran_array_wrappable<CppType>::value,
                        int>::type = 0>
                std::string operator()() const {
                    return fortran_param_type_from_c_f{}.template operator()<CType>();
                }
            };

            template <typename CSignature>
            struct has_array_descriptor
                : is_there_in_sequence_if<typename boost::function_types::parameter_types<CSignature>::type,
                      std::is_same<boost::mpl::_, gt_fortran_array_descriptor *>> {};

            /**
             * @brief This function writes the `interface`-section of the fortran-code.
             * @param strm Stream, where the output will be written to
             * @param c_name The name of the function in the c-header
             * @param fortran_name The name of the function in the c-bindings of the module.
             */
            template <class CSignature>
            std::ostream &write_fortran_binding(std::ostream &strm, char const *c_name, char const *fortran_name) {
                namespace ft = boost::function_types;
                std::stringstream tmp_strm;
                tmp_strm << fortran_return_type<typename ft::result_type<CSignature>::type>() << " " << fortran_name
                         << "(";
                for_each_param<CSignature>(ignore_type_f{}, [&](const std::string &, int i) {
                    if (i)
                        tmp_strm << ", ";
                    tmp_strm << "arg" << i;
                });
                tmp_strm << ")";
                if (strcmp(c_name, fortran_name) == 0)
                    tmp_strm << " bind(c)";
                else
                    tmp_strm << " bind(c, name=\"" << c_name << "\")";
                strm << wrap_line(tmp_strm.str(), "    ");
                strm << "      use iso_c_binding\n";
                if (has_array_descriptor<CSignature>::value)
                    strm << "      use array_descriptor\n";
                for_each_param<CSignature>(fortran_param_type_from_c_f{}, [&](const std::string &type_name, int i) {
                    strm << "      " << type_name << " :: arg" << i << "\n";
                });
                return strm << "    end "
                            << fortran_function_specifier<typename ft::result_type<CSignature>::type>() + "\n";
            }

            struct cpp_type_descriptor_f {
                template <class CppType,
                    class CType = param_converted_to_c_t<CppType>,
                    typename std::enable_if<std::is_same<CType, gt_fortran_array_descriptor *>::value &&
                                                is_fortran_array_wrappable<CppType>::value,
                        int>::type = 0>
                boost::optional<gt_fortran_array_descriptor> operator()() const {
                    static const gt_fortran_array_descriptor meta =
                        get_fortran_view_meta((add_pointer_t<CppType>){nullptr});
                    return meta;
                }
                template <class CppType,
                    class CType = param_converted_to_c_t<CppType>,
                    typename std::enable_if<!std::is_same<CType, gt_fortran_array_descriptor *>::value ||
                                                !is_fortran_array_wrappable<CppType>::value,
                        int>::type = 0>
                boost::optional<gt_fortran_array_descriptor> operator()() const {
                    return boost::none;
                }
            };
            /**
             * @brief This function writes the `contains`-section of the fortran-code.
             * @param strm Stream, where the output will be written to
             * @param fortran_cbindings_name The name of the function in the c-bindings-part of the module.
             * @param fortran_name The name of the function in the fortran-part of the module.
             */
            template <class CppSignature>
            std::ostream &write_fortran_wrapper(
                std::ostream &strm, char const *fortran_cbindings_name, const char *fortran_name) {
                using CSignature = wrapped_t<CppSignature>;
                namespace ft = boost::function_types;

                std::stringstream tmp_strm;
                tmp_strm << fortran_return_type<typename ft::result_type<CSignature>::type>() << " " << fortran_name
                         << "(";
                for_each_param<CSignature>(ignore_type_f{}, [&](const std::string &, int i) {
                    if (i)
                        tmp_strm << ", ";
                    tmp_strm << "arg" << i;
                });
                tmp_strm << ")";
                strm << wrap_line(tmp_strm.str(), "    ");

                strm << "      use iso_c_binding\n";
                if (has_array_descriptor<CSignature>::value) {
                    strm << "      use array_descriptor\n";
                }
                for_each_param<CppSignature>(fortran_param_type_from_cpp_f{}, [&](const std::string &type_name, int i) {
                    strm << "      " << type_name << ", target :: arg" << i << "\n";
                });

                for_each_param<CppSignature>(
                    cpp_type_descriptor_f{}, [&](const boost::optional<gt_fortran_array_descriptor> &meta, int i) {
                        if (meta) {
                            const auto desc_name = "descriptor" + std::to_string(i);
                            strm << "      type(gt_fortran_array_descriptor) :: " + desc_name + "\n";
                        }
                    });
                strm << "\n";

                for_each_param<CppSignature>(
                    cpp_type_descriptor_f{}, [&](const boost::optional<gt_fortran_array_descriptor> &meta, int i) {
                        if (meta) {
                            const auto var_name = "arg" + std::to_string(i);
                            const auto desc_name = "descriptor" + std::to_string(i);
                            std::string c_loc = "c_loc(" + var_name + "(";
                            for (int i = 0; i < meta->rank; ++i) {
                                if (i)
                                    c_loc += ",";
                                c_loc += "lbound(" + var_name + ", " + std::to_string(i + 1) + ")";
                            }
                            c_loc += "))";
                            if (meta->is_acc_present)
                                strm << "      !$acc data present(" << var_name << ")\n" //
                                     << "      !$acc host_data use_device(" << var_name << ")\n";

                            strm << "      " << desc_name << "%rank = " << meta->rank << "\n"                 //
                                 << "      " << desc_name << "%type = " << meta->type << "\n"                 //
                                 << "      " << desc_name << "%dims = reshape(shape(" << var_name << "), &\n" //
                                 << "        shape(" << desc_name << "%dims), (/0/))\n"                       //
                                 << "      " << desc_name << "%data = " << c_loc << "\n";
                            if (meta->is_acc_present)
                                strm << "      !$acc end host_data\n" //
                                     << "      !$acc end data\n";
                            strm << "\n";
                        }
                    });

                tmp_strm.str("");
                if (std::is_void<typename ft::result_type<CSignature>::type>::value) {
                    tmp_strm << "call " << fortran_cbindings_name << "(";
                } else {
                    tmp_strm << fortran_name << " = " << fortran_cbindings_name << "(";
                }
                for_each_param<CppSignature>(
                    cpp_type_descriptor_f{}, [&](const boost::optional<gt_fortran_array_descriptor> &meta, int i) {
                        if (i)
                            tmp_strm << ", ";
                        if (meta) {
                            const auto desc_name = "descriptor" + std::to_string(i);
                            tmp_strm << desc_name;
                        } else {
                            tmp_strm << "arg" << i;
                        }
                    });
                tmp_strm << ")";
                strm << wrap_line(tmp_strm.str(), "      ");

                return strm << "    end "
                            << fortran_function_specifier<typename ft::result_type<CSignature>::type>() + "\n";
            }

            struct c_bindings_traits {
                template <class CSignature>
                static void generate_entity(std::ostream &strm, char const *c_name) {
                    write_c_binding<CSignature>(strm, c_name);
                }
            };

            struct fortran_bindings_traits {
                template <class CSignature>
                static void generate_entity(std::ostream &strm, char const *c_name, char const *fortran_name) {
                    write_fortran_binding<CSignature>(strm, c_name, fortran_name);
                }
            };

            struct fortran_wrapper_traits {
                template <class CppSignature>
                static void generate_entity(
                    std::ostream &strm, char const *fortran_cbindings_name, const char *fortran_name) {
                    write_fortran_wrapper<CppSignature>(strm, fortran_cbindings_name, fortran_name);
                }
            };

            template <class Traits, class Signature, class... Params>
            void add_entity(const char *name, Params &&... params) {
                get_entities<Traits>().add(name,
                    std::bind(Traits::template generate_entity<Signature>,
                        std::placeholders::_1,
                        std::forward<Params>(params)...));
            }

            template <class CSignature>
            struct registrar_simple {
                registrar_simple(char const *name) {
                    add_entity<_impl::c_bindings_traits, CSignature>(name, name);
                    add_entity<_impl::fortran_bindings_traits, CSignature>(name, name, name);
                }
            };
            template <class CppSignature>
            struct registrar_wrapped {
                registrar_wrapped(char const *c_name, char const *fortran_cbindings_name, char const *fortran_name) {
                    using CSignature = wrapped_t<CppSignature>;
                    add_entity<_impl::c_bindings_traits, CSignature>(c_name, c_name);
                    add_entity<_impl::fortran_bindings_traits, CSignature>(c_name, c_name, fortran_cbindings_name);
                    add_entity<_impl::fortran_wrapper_traits, CppSignature>(
                        c_name, fortran_cbindings_name, fortran_name);
                }
            };

            struct fortran_generic_registrar {
                fortran_generic_registrar(char const *generic_name, char const *concrete_name);
            };
        } // namespace _impl

        /// Outputs the content of the C compatible header with the declarations added by GT_ADD_GENERATED_DECLARATION
        void generate_c_interface(std::ostream &strm);

        /// Outputs the content of the Fortran module with the declarations added by GT_ADD_GENERATED_DECLARATION
        void generate_fortran_interface(std::ostream &strm, std::string const &module_name);
    } // namespace c_bindings
} // namespace gridtools

/**
 *  Registers the function that for declaration generations.
 *  Users should not use these directly.
 */
#define GT_ADD_GENERATED_DECLARATION(csignature, name) \
    static ::gridtools::c_bindings::_impl::registrar_simple<csignature> generated_declaration_registrar_##name(#name)
#define GT_ADD_GENERATED_DECLARATION_WRAPPED(cppsignature, name)                                                   \
    static ::gridtools::c_bindings::_impl::registrar_wrapped<cppsignature> generated_declaration_registrar_##name( \
        #name, BOOST_PP_STRINGIZE(BOOST_PP_CAT(name, _impl)), #name)

#define GT_ADD_GENERIC_DECLARATION(generic_name, concrete_name)      \
    static ::gridtools::c_bindings::_impl::fortran_generic_registrar \
        fortran_generic_registrar_##generic_name##_##concrete_name(#generic_name, #concrete_name)
