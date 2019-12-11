/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <exception>
#include <string>
#include <type_traits>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/seq/transform.hpp>
#include <boost/preprocessor/seq/variadic_seq_to_seq.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

#include <cpp_bindgen/export.hpp>

#include "../common/defs.hpp"
#include "../meta/id.hpp"
#include "../storage/data_store.hpp"
#include "fortran_array_adapter.hpp"

namespace repository_impl_ {
    template <class T>
    T const &try_return(typename ::gridtools::meta::lazy::id<T>::type const &arg, char const *msg) {
        return arg;
    }
    template <class T, class U>
    T const &try_return(U const &arg, char const *msg) {
        throw std::runtime_error(msg);
    }

    template <class...>
    struct ctor_signature : std::false_type {};

    template <class... Rs, class... Args>
    struct ctor_signature<Rs(Args...)...> : std::true_type {
        using args = ::gridtools::meta::list<Args...>;
    };

    template <class Repo, class T, T const Repo::*Field>
    void set_field(Repo const &repo, ::gridtools::fortran_array_adapter<T> view) {
        view.transform_to(repo.*Field);
    }
} // namespace repository_impl_

#define GT_REPO_DATA_STORE(item) BOOST_PP_TUPLE_ELEM(2, 0, item)
#define GT_REPO_DATA_STORE_NAME(item) BOOST_PP_STRINGIZE(GT_REPO_DATA_STORE(item))
#define GT_REPO_DATA_STORE_TYPE(item) BOOST_PP_CAT(GT_REPO_DATA_STORE(item), _t)
#define GT_REPO_MAKE(item) BOOST_PP_CAT(GT_REPO_DATA_STORE(item), _make)

#define GT_REPO_FIELD_VAR(field) BOOST_PP_TUPLE_ELEM(2, 1, field)
#define GT_REPO_FIELD_NAME(field) BOOST_PP_STRINGIZE(GT_REPO_FIELD_VAR(field))

#define GT_REPO_BUILDER(builder) BOOST_PP_TUPLE_ELEM(2, 1, builder)

#define GT_REPO_BUILDER_TYPE_HELPER(r, _, builder) decltype(GT_REPO_BUILDER(builder))

#define GT_REPO_DEFINE_ALIAS_HELPER(r, Args, builder)                                                       \
    using GT_REPO_DATA_STORE_TYPE(builder) = decltype(GT_REPO_BUILDER(builder)(std::declval<Args>()...)()); \
    static_assert(::gridtools::storage::is_data_store_ptr<GT_REPO_DATA_STORE_TYPE(builder)>::value,         \
        "builders should return data_stores");

#define GT_REPO_DEFINE_MAKE_HELPER(r, Args, builder)                           \
    static auto GT_REPO_MAKE(builder)(std::string const &name, Args... args) { \
        return GT_REPO_BUILDER(builder)(args...).name(name)();                 \
    }

#define GT_REPO_DEFINE_FIELD_HELPER(r, _, field) GT_REPO_DATA_STORE_TYPE(field) const GT_REPO_FIELD_VAR(field);

#define GT_REPO_CONSTRUCT_FIELD_HELPER(r, args, field) \
    GT_REPO_FIELD_VAR(field)(GT_REPO_MAKE(field)(GT_REPO_FIELD_NAME(field), args...))

#define GT_REPO_TRY_RETURN_HELPER(r, type_and_name, field)                                                        \
    if (BOOST_PP_TUPLE_ELEM(2, 1, type_and_name) == GT_REPO_FIELD_NAME(field))                                    \
        return ::repository_impl_::try_return<BOOST_PP_TUPLE_ELEM(2, 0, type_and_name)>(GT_REPO_FIELD_VAR(field), \
            "type mismatch: " GT_REPO_FIELD_NAME(field) " is " GT_REPO_DATA_STORE_NAME(field) " field");

#define GT_REPO_DEFINE_MAP_HELPER(r, _, builder)                                \
    decltype(auto) GT_REPO_DATA_STORE(builder)(std::string const &name) const { \
        return field<GT_REPO_DATA_STORE_TYPE(builder)>(name);                   \
    }

#define GT_REPO_CALL_FUN_HELPER(r, fun, field) fun(GT_REPO_FIELD_VAR(field));

#define GT_REPO_DEFINE_REPOSITORY(class_name, builders, fields)                                              \
    namespace class_name##_impl_ {                                                                           \
        using ctor_signature = ::repository_impl_::ctor_signature<BOOST_PP_SEQ_ENUM(                         \
            BOOST_PP_SEQ_TRANSFORM(GT_REPO_BUILDER_TYPE_HELPER, , builders))>;                               \
        static_assert(ctor_signature::value, "all builders should have the same signature");                 \
        template <class Args = typename ctor_signature::args>                                                \
        struct repo;                                                                                         \
        template <template <class...> class L, class... Args>                                                \
        struct repo<L<Args...>> {                                                                            \
            BOOST_PP_SEQ_FOR_EACH(GT_REPO_DEFINE_MAKE_HELPER, Args, builders)                                \
          public:                                                                                            \
            BOOST_PP_SEQ_FOR_EACH(GT_REPO_DEFINE_ALIAS_HELPER, Args, builders)                               \
            BOOST_PP_SEQ_FOR_EACH(GT_REPO_DEFINE_FIELD_HELPER, , fields)                                     \
            repo(Args... args)                                                                               \
                : BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_TRANSFORM(GT_REPO_CONSTRUCT_FIELD_HELPER, args, fields)) {} \
            template <class T>                                                                               \
            decltype(auto) field(std::string const &name) const {                                            \
                BOOST_PP_SEQ_FOR_EACH(GT_REPO_TRY_RETURN_HELPER, (T, name), fields)                          \
                throw std::runtime_error(name + " is not found in the repository");                          \
            }                                                                                                \
            BOOST_PP_SEQ_FOR_EACH(GT_REPO_DEFINE_MAP_HELPER, _, builders)                                    \
            template <class Fun>                                                                             \
            void for_each(Fun const &fun) const {                                                            \
                BOOST_PP_SEQ_FOR_EACH(GT_REPO_CALL_FUN_HELPER, fun, fields)                                  \
            }                                                                                                \
        };                                                                                                   \
    }                                                                                                        \
    using class_name = class_name##_impl_::repo<>

#define GT_REPO_BINDING_IMPL(repo, field) (::repository_impl_::set_field<repo, decltype(repo::field), &repo::field>)

#define GT_REPO_DEFINE_BINDING_HELPER(r, data, field)                                      \
    BINDGEN_EXPORT_BINDING_WRAPPED_2(BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(2, 1, data), field), \
        GT_REPO_BINDING_IMPL(BOOST_PP_TUPLE_ELEM(2, 0, data), field));

#define GT_REPO_DEFINE_REPOSITORY_BINDINGS(name, prefix, fields) \
    BOOST_PP_SEQ_FOR_EACH(GT_REPO_DEFINE_BINDING_HELPER, (name, prefix), fields) static_assert(1, "")

/*
 * @brief entry for the user
 * @param name class name for the repository
 * @param builders BOOST_PP sequence of tuples of the form (data_store_type_name, data_store_builder_generator)
 * @param fields BOOST_PP sequence of tuples of the form (data_store_type_name, field_name)
 *
 * All data store builder generators must be the functions with exactly the same arguments.
 * The constructor of generated type accepts the same arguments that the builder generators are.
 *
 * Say there are functions ijk_builder(int, int) and ij_builder(int, int).
 * For the macro invocation `GT_DEFINE_REPOSITORY(repo, (ijk ijk_builder)(ij ij_builder), (ijk u)(ijk v)(ij crlat))`
 * Synopsis of the generated class is:
 * ```
 *   class repo {
 *   public:
 *     // aliases for the filed types
 *     using ijk_t = decltype(ijk_builder(0,0).name("")());
 *     using ij_t = decltype(ij_builder(0,0).name("")());
 *
 *     // field public members for direct access
 *     ijk_t const u;
 *     ijk_t const v;
 *     ij_t const crlat;
 *
 *     // ctor delegates to the builders; additionally the name is set for each field ("u" for u, "v" for v etc. )
 *     repo(int, int);
 *
 *     // get the field of type `T` by name.
 *     template <class T> T const& field(std::string const&) const;
 *
 *     // get `ijk` field by name
 *     ijk_t ijk(std::string const&) const;
 *     // get `ij` field by name
 *     ij_t ij(std::string const&) const;
 *
 *     // call function `f` with each field in the repo
 *     template <class F> for_each(F const& f) const;
 *   };
 * ```
 *
 * Main macro is GT_REPO_DEFINE_REPOSITORY. Here we just add extra parenthesis to the input to make user-code
 * look nicer (no double parenthesis)
 */
#define GT_DEFINE_REPOSITORY(name, builders, fields) \
    GT_REPO_DEFINE_REPOSITORY(name, BOOST_PP_VARIADIC_SEQ_TO_SEQ(builders), BOOST_PP_VARIADIC_SEQ_TO_SEQ(fields))

/*
 * @brief Creates the fortran bindings for the repository. Must be called from a cpp file.
 * @param name class name for the repository. Should be generated by GT_DEFINE_REPOSITORY
 * @param prefix prefix that will be appended to bindings
 * @param ...  fields
 *
 * Suppose you have a repository with name = "CRep", fortran_name = "FRep", fields named "u" and "v" and prefix set
 * to "prefix_". This will generate the following fortran bindings:
 *      prefix_u(repo, arr) sets CRep::u
 *      prefix_v(repo, arr) sets CRep::v
 */
#define GT_DEFINE_REPOSITORY_BINDINGS(name, prefix, ...)                                                        \
    BOOST_PP_SEQ_FOR_EACH(GT_REPO_DEFINE_BINDING_HELPER, (name, prefix), BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)) \
    static_assert(1, "")
