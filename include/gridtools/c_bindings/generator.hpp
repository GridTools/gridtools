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
#include <string>
#include <vector>

#include <boost/function_types/result_type.hpp>
#include <boost/function_types/parameter_types.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/type_index.hpp>

#include <common/generic_metafunctions/is_there_in_sequence_if.hpp>

#include "function_wrapper.hpp"

namespace gridtools {
    namespace c_bindings {

        namespace _impl {

            struct c_string_less {
                bool operator()(char const *lhs, char const *rhs) const { return strcmp(lhs, rhs) < 0; }
            };

            class declarations {
                using generator_t = std::function< void(std::ostream &, char const *) >;
                std::map< char const *, generator_t, c_string_less > m_generators;

              public:
                void add(char const *name, generator_t generator);
                friend std::ostream &operator<<(std::ostream &strm, declarations const &obj);
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

            struct get_c_type_name_f {
                template < class T >
                std::string operator()() const {
                    return boost::typeindex::type_id< typename recursive_remove_cv< T >::type >().pretty_name();
                }
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
                template < class Fun, class TypeToStr, class T >
                void operator()(Fun &&fun, TypeToStr &&type_to_str, int &count, boxed< T >) const {
                    std::forward< Fun >(fun)(type_to_str.template operator() < T > (), count);
                    ++count;
                }
            };

            template < class Signature, class TypeToStr, class Fun >
            void for_each_param(TypeToStr &&type_to_str, Fun &&fun) {
                namespace m = boost::mpl;
                int count = 0;
                m::for_each< typename boost::function_types::parameter_types< Signature >::type, boxed< m::_ > >(
                    std::bind(apply_to_param_f{},
                        std::forward< Fun >(fun),
                        std::forward< TypeToStr >(type_to_str),
                        std::ref(count),
                        std::placeholders::_1));
            };

            template < class CSignature >
            std::ostream &write_c_declaration(std::ostream &strm, char const *name) {
                namespace ft = boost::function_types;
                strm << get_c_type_name< typename ft::result_type< CSignature >::type >() << " " << name << "(";
                for_each_param< CSignature >(get_c_type_name_f{},
                    [&](const std::string &type_name, int i) {
                        if (i)
                            strm << ", ";
                        strm << type_name;
                    });
                return strm << ");\n";
            }

            template < class >
            struct fortran_kind_name {
                static char const value[];
            };

            template <>
            char const fortran_kind_name< bool >::value[];
            template <>
            char const fortran_kind_name< int >::value[];
            template <>
            char const fortran_kind_name< short >::value[];
            template <>
            char const fortran_kind_name< long >::value[];
            template <>
            char const fortran_kind_name< long long >::value[];
            template <>
            char const fortran_kind_name< float >::value[];
            template <>
            char const fortran_kind_name< double >::value[];
            template <>
            char const fortran_kind_name< long double >::value[];
            template <>
            char const fortran_kind_name< signed char >::value[];

            template < class T,
                typename std::enable_if< std::is_same< typename std::decay< T >::type, bool >::value, int >::type = 0 >
            std::string fortran_type_name() {
                using decayed_t = typename std::decay< T >::type;
                return std::string("logical(") + fortran_kind_name< decayed_t >::value + ")";
            }
            template < class T,
                typename std::enable_if< std::is_integral< T >::value &&
                                             !std::is_same< typename std::decay< T >::type, bool >::value,
                    int >::type = 0 >
            std::string fortran_type_name() {
                using signed_decayed_t = typename std::make_signed< typename std::decay< T >::type >::type;
                return std::string("integer(") + fortran_kind_name< signed_decayed_t >::value + ")";
            }

            template < class T, typename std::enable_if< std::is_floating_point< T >::value, int >::type = 0 >
            std::string fortran_type_name() {
                using decayed_t = typename std::decay< T >::type;
                return std::string("real(") + fortran_kind_name< decayed_t >::value + ")";
            }

            template < class T, typename std::enable_if< std::is_pointer< T >::value, int >::type = 0 >
            std::string fortran_type_name() {
                return "type(c_ptr)";
            }

            template < class T,
                typename std::enable_if< !std::is_pointer< T >::value && !std::is_integral< T >::value &&
                                             !std::is_floating_point< T >::value,
                    int >::type = 0 >
            std::string fortran_type_name() {
                assert("Unsupported fortran type." && false);
                return "";
            }

            template < class T, typename std::enable_if< std::is_void< T >::value, int >::type = 0 >
            std::string fortran_return_type() {
                return "subroutine";
            }

            template < class T, typename std::enable_if< !std::is_void< T >::value, int >::type = 0 >
            std::string fortran_return_type() {
                return fortran_type_name< T >() + " function";
            }

            struct ignore_type_f {
                template < class T >
                std::string operator()() const {
                    return "";
                }
            };

            enum class fortran_param_style { c_bindings, indirection };
            struct dimensions_f {
                template < class N >
                void operator()(N, std::string &s) const {
                    s += (N::value == 0 ? ":" : ",:");
                }
            };
            struct fortran_param_type_common_f {
                template < class CType,
                    typename std::enable_if<
                        !std::is_same< CType, gt_fortran_array_descriptor >::value &&
                            (!std::is_pointer< CType >::value ||
                                std::is_class< typename std::remove_pointer< CType >::type >::value),
                        int >::type = 0 >
                std::string operator()() const {
                    return fortran_type_name< CType >() + ", value";
                }

                template < class CType,
                    typename std::enable_if<
                        std::is_pointer< CType >::value &&
                            std::is_arithmetic< typename std::remove_pointer< CType >::type >::value,
                        int >::type = 0 >
                std::string operator()() const {
                    return fortran_type_name< typename std::remove_pointer< CType >::type >() + ", dimension(*)";
                }
                template < class CType,
                    typename std::enable_if<
                        std::is_pointer< CType >::value &&
                            !std::is_arithmetic< typename std::remove_pointer< CType >::type >::value &&
                            !std::is_class< typename std::remove_pointer< CType >::type >::value,
                        int >::type = 0 >
                std::string operator()() const {
                    return "type(c_ptr)";
                }
            };
            template < fortran_param_style style >
            struct fortran_param_type_f;

            template <>
            struct fortran_param_type_f< fortran_param_style::c_bindings > {

                template < class CType,
                    typename std::enable_if< std::is_same< CType, gt_fortran_array_descriptor >::value, int >::type =
                        0 >
                std::string operator()() const {
                    return "type(gt_fortran_array_descriptor), value";
                }

                template < class CType,
                    typename std::enable_if< !std::is_same< CType, gt_fortran_array_descriptor >::value, int >::type =
                        0 >
                std::string operator()() const {
                    return fortran_param_type_common_f{}.template operator()< CType >();
                }
            };
            template <>
            struct fortran_param_type_f< fortran_param_style::indirection > {

                template < class CppType,
                    class CType = param_converted_to_c_t< CppType >,
                    typename std::enable_if< std::is_same< CType, gt_fortran_array_descriptor >::value, int >::type =
                        0 >
                std::string operator()() const {
                    using indices = meta::make_indices< fortran_array_view_rank< CppType >::value >;
                    std::string dimensions = "dimension(";
                    for_each< indices >(std::bind(dimensions_f{}, std::placeholders::_1, std::ref(dimensions)));
                    dimensions += ")";
                    return fortran_type_name< typename fortran_array_view_element_type< CppType >::type >() + ", " +
                           dimensions;
                }

                template < class CppType,
                    class CType = param_converted_to_c_t< CppType >,
                    typename std::enable_if< !std::is_same< CType, gt_fortran_array_descriptor >::value, int >::type =
                        0 >
                std::string operator()() const {
                    return fortran_param_type_common_f{}.template operator()< CType >();
                }
            };

            template < class CSignature >
            std::ostream &write_fortran_cbindings_declaration(std::ostream &strm, char const *name) {
                namespace ft = boost::function_types;
                constexpr bool has_array_descriptor =
                    is_there_in_sequence_if< typename ft::parameter_types< CSignature >::type,
                        std::is_same< boost::mpl::_, gt_fortran_array_descriptor > >::value;
                strm << "    " << fortran_return_type< typename ft::result_type< CSignature >::type >() << " " << name
                     << "(";
                for_each_param< CSignature >(ignore_type_f{},
                    [&](const std::string &, int i) {
                        if (i)
                            strm << ", ";
                        strm << "arg" << i;
                    });
                strm << ") bind(c)\n      use iso_c_binding\n";
                if (has_array_descriptor) {
                    strm << "      use array_descriptor\n";
                }
                for_each_param< CSignature >(fortran_param_type_f< fortran_param_style::c_bindings >{},
                    [&](const std::string &type_name, int i) {
                        strm << "      " << type_name << " :: arg" << i << "\n";
                    });
                return strm << "    end\n";
            }

            struct argument_wrapper_f {
                template < class T,
                    typename std::enable_if< std::is_same< T, gt_fortran_array_descriptor >::value, int >::type = 0 >
                std::pair< std::string, std::string > operator()() const {
                    return std::make_pair("create_array_descriptor(", ")");
                }
                template < class T,
                    typename std::enable_if< !std::is_same< T, gt_fortran_array_descriptor >::value, int >::type = 0 >
                std::pair< std::string, std::string > operator()() const {
                    return std::make_pair("", "");
                }
            };
            template < class CppSignature >
            std::ostream &write_fortran_indirection_declaration(
                std::ostream &strm, char const *c_name, const char *fortran_name) {
                using CSignature = wrapped_t< CppSignature >;
                namespace ft = boost::function_types;
                constexpr bool has_array_descriptor =
                    is_there_in_sequence_if< typename ft::parameter_types< CSignature >::type,
                        std::is_same< boost::mpl::_, gt_fortran_array_descriptor > >::value;

                strm << "    " << fortran_return_type< typename ft::result_type< CSignature >::type >() << " "
                     << fortran_name << "(";
                for_each_param< CSignature >(ignore_type_f{},
                    [&](const std::string &, int i) {
                        if (i)
                            strm << ", ";
                        strm << "arg" << i;
                    });
                strm << ") bind(c)\n      use iso_c_binding\n";
                if (has_array_descriptor) {
                    strm << "      use array_descriptor\n";
                }
                for_each_param< CppSignature >(fortran_param_type_f< fortran_param_style::indirection >{},
                    [&](const std::string &type_name, int i) {
                        strm << "      " << type_name << " :: arg" << i << "\n";
                    });
                if (std::is_void< typename ft::result_type< CSignature >::type >::value) {
                    strm << "      call " << c_name << "(";
                } else {
                    strm << "      " << fortran_name << " = " << c_name << "(";
                }
                for_each_param< CSignature >(argument_wrapper_f{},
                    [&](const std::pair< std::string, std::string > &arg_wrapper, int i) {
                        if (i)
                            strm << ", ";
                        strm << arg_wrapper.first << "arg" << i << arg_wrapper.second;
                    });
                strm << ")\n";

                return strm << "    end\n";
            }

            struct c_traits {
                template < class CSignature >
                static void generate_declaration(std::ostream &strm, char const *c_name) {
                    write_c_declaration< CSignature >(strm, c_name);
                }
            };

            struct fortran_cbindings_traits {
                template < class CSignature >
                static void generate_declaration(std::ostream &strm, char const *c_name) {
                    write_fortran_cbindings_declaration< CSignature >(strm, c_name);
                }
            };

            struct fortran_indirection_traits {
                template < class CppSignature >
                static void generate_declaration(std::ostream &strm, char const *c_name, const char *fortran_name) {
                    write_fortran_indirection_declaration< CppSignature >(strm, c_name, fortran_name);
                }
            };

            template < class Traits, class Signature, class... Params >
            void add_declaration(const char *name, Params &&... params) {
                get_declarations< Traits >().add(name,
                    std::bind(Traits::template generate_declaration< Signature >,
                                                     std::placeholders::_1,
                                                     std::forward< Params >(params)...));
            }

            template < class CSignature >
            struct registrar_simple {
                registrar_simple(char const *name) {
                    add_declaration< _impl::c_traits, CSignature >(name, name);
                    add_declaration< _impl::fortran_cbindings_traits, CSignature >(name, name);
                }
            };
            template < class CppSignature >
            struct registrar_extended {
                registrar_extended(char const *c_name, char const *fortran_name) {
                    using CSignature = wrapped_t< CppSignature >;
                    add_declaration< _impl::c_traits, CSignature >(c_name, c_name);
                    add_declaration< _impl::fortran_cbindings_traits, CSignature >(c_name, c_name);
                    add_declaration< _impl::fortran_indirection_traits, CppSignature >(c_name, c_name, fortran_name);
                }
            };

            struct fortran_generic_registrar {
                fortran_generic_registrar(char const *generic_name, char const *concrete_name);
            };
        }

        /// Outputs the content of the C compatible header with the declarations added by GT_ADD_GENERATED_DECLARATION
        void generate_c_interface(std::ostream &strm);

        /// Outputs the content of the Fortran module with the declarations added by GT_ADD_GENERATED_DECLARATION
        void generate_fortran_interface(std::ostream &strm, std::string const &module_name);
    }
}

/**
 *  Registers the function that for declaration generations.
 *  Users should not this directly.
 */
#define GT_ADD_GENERATED_DECLARATION(csignature, name) \
    static ::gridtools::c_bindings::_impl::registrar_simple< csignature > generated_declaration_registrar_##name(#name)
#define GT_ADD_GENERATED_DECLARATION_EX(cppsignature, name)                                                           \
    static ::gridtools::c_bindings::_impl::registrar_extended< cppsignature > generated_declaration_registrar_##name( \
        BOOST_PP_STRINGIZE(BOOST_PP_CAT(name, _impl)), #name)

#define GT_ADD_GENERIC_DECLARATION(generic_name, concrete_name)      \
    static ::gridtools::c_bindings::_impl::fortran_generic_registrar \
        fortran_generic_registrar_##generic_name##_##concrete_name(#generic_name, #concrete_name)
