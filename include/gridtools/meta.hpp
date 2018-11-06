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

/**
 *  C++11 metaprogramming library.
 *
 *  Basic Concepts
 *  ==============
 *
 *  List
 *  ----
 *  An instantiation of the template class with class template parameters.
 *
 *  Examples of lists:
 *    meta::list<void, int> : elements are void and int
 *    std::tuple<double, double> : elements are double and double
 *    std::vector<std::tuple<>, some_allocator>: elements are std::tuple<> and some_allocator
 *
 *  Examples of non lists:
 *    std::array<N, double> : first template argument is not a class
 *    int : is not the instantiation of template
 *    struct foo; is not an instantiation of template
 *
 *  Function
 *  --------
 *  A template class or an alias with class template parameters.
 *  Note the difference with MPL approach: function is not required to have `type` inner alias.
 *  Functions that have `type` inside are called lazy functions in the context of this library.
 *  The function arguments are the actual parameters of the instantiation: Arg1, Arg2 etc. in F<Arg1, Arg2 etc.>
 *  The function invocation result is just F<Arg1, Arg2 etc.> not F<Arg1, Arg2 etc.>::type.
 *  This simplification of the function concepts (comparing with MPL) is possible because of C++ aliases.
 *  And it is significant for compile time performance.
 *
 *  Examples of functions:
 *    - std::is_same
 *    - std::pair
 *    - std::tuple
 *    - meta::list
 *    - meta::is_list
 *
 *  Examples of non functions:
 *    - std::array : first parameter is not a class
 *    - meta::list<int> : is not a template
 *
 *  In the library some functions have integers as arguments. Usually they have `_c` suffix and have the sibling
 *  without prefix. Disadvantage of having such a hybrid signature, that those functions can not be passed as
 *  arguments to high order functions.
 *
 *  Meta Class
 *  ----------
 *  A class that have `apply` inner template class or alias, which is a function [here and below the term `function`
 *  used in the context of this library]. Meta classes are used to return functions from functions.
 *
 *  Examples:
 *    - meta::always<void>
 *    - meta::rename<std::tuple>
 *
 *  High Order Function
 *  -------------------
 *  A template class or alias which first parameters are template of class class templates and the rest are classes
 *  Examples of metafuction signatures:
 *  template <template <class...> class, class...> struct foo;
 *  template <template <class...> class, template <class...> class> struct bar;
 *  template <template <class...> class...> struct baz;
 *
 *  Examples:
 *    - meta::rename
 *    - meta::lfold
 *    - meta::is_instantiation_of
 *
 *  Library Structure
 *  =================
 *
 *  It consists of the set of functions, `_c` functions and high order functions.
 *
 *  Regularly, a function has also its lazy version, which is defined in the `lazy` nested namespace under the same
 *  name. Exceptions are functions that return:
 *   - a struct with a nested `type` alias, which points to the struct itself;
 *       ex: `list`
 *   - a struct derived from `std::intergral_constant`
 *       ex: `length`, `is_list`
 *   - meta class
 *
 *  nVidia and Intel compilers with versions < 9 and < 18 respectively have a bug that doesn't allow to use template
 *  aliases. To deal with that, the library has two modes that are switching by `GT_BROKEN_TEMPLATE_ALIASES` macro.
 *  If the value of `GT_BROKEN_TEMPLATE_ALIASES` is set to non zero, the notion of function is degradated to lazy
 *  function like in MPL.
 *
 *  In this case non-lazy functions don't exist and `lazy` nested namespace is `inline` [I.e. `meta::concat`
 *  for example is the same as `meta::lazy::concat`]. High order functions in this case interpret their functional
 *  parameters as a lazy functions [I.e. they use `::type` to invoke them].
 *
 *  `GT_META_CALL` and `GT_META_DEFINE_ALIAS` macros are defined to help keep the user code independent on that
 *  interface difference. Unfortunately in general case, it is not always possible to maintain that compatibility
 *  only using that two macros. Direct <tt>\#if GT_BROKEN_TEMPLATE_ALIASES`</tt> could be necessary.
 *
 *  Syntax sugar: All high order functions being called with only functional arguments return partially applied
 *  versions of themselves [which became plane functions].
 *  Example, where it could be useful is:
 *  transform a list of lists:  <tt>using out = meta::transform<meta::transform<fun>::apply, in>;</tt>
 *
 *  Guidelines for Using Meta in Compatible with Retarded Compilers Mode
 *  =====================================================================
 *    - don't punic;
 *    - write and debug your code for some sane compiler pretending that template aliases are not a problem;
 *    - uglify each and every call of the function from meta `namespace` with `GT_META_CALL` macro;
 *      for example the code like:
 *         using my_stuff = meta::concat<a, meta::front<b>, meta::clear<c>>;
 *      should be uglified like:
 *         using m_staff = GT_META_CALL(meta::concat, (GT_META_CALL(meta::front, a), GT_META_CALL(meta::clear, c)));
 *    - uglify with the same macro calls to the functions that you define using composition of `meta::` functions;
 *    - replace every definition of template alias in you code with `GT_META_DEFINE_ALIAS`;
 *      for example the code like:
 *         template <class T, class U>
 *         using my_lookup = meta::second<meta::mp_find<typename T::the_map, my_get_key<U>>>;
 *      should be uglified like:
 *         template <class T, class U>
 *         GT_META_DEFINE_ALIAS(my_lookup, meta::second, (GT_META_CALL(meta::mp_find,
 *            (GT_META_CALL(typename T::the_map, GT_META_CALL(my_get_key, U)))));
 *    - modifications above should not break compilation for the sane compiler, check it;
 *    - also check if the code compiles for your retarded compiler;
 *    - if yes, you are lucky;
 *    - if not, possible reason is that you have hand written lazy function and its `direct` counterpart that is
 *      defined smth. like `template <class T> using foo = lazy_foo<T>;` and you pass `foo` to the high order
 * function.
 *      in this case, you need to add retarded version (where `lazy_foo` would just named `foo`) under
 *      <tt>\#if GT_BROKEN_TEMPLATE_ALIASES</tt>;
 *    - if it is still not your case, ask \@anstaf.
 *
 *  TODO List
 *  =========
 *   - add numeric stuff like `plus`, `less` etc.
 */

#include "meta/always.hpp"
#include "meta/at.hpp"
#include "meta/bind.hpp"
#include "meta/cartesian_product.hpp"
#include "meta/clear.hpp"
#include "meta/combine.hpp"
#include "meta/concat.hpp"
#include "meta/ctor.hpp"
#include "meta/curry.hpp"
#include "meta/curry_fun.hpp"
#include "meta/dedup.hpp"
#include "meta/defer.hpp"
#include "meta/defs.hpp"
#include "meta/drop_back.hpp"
#include "meta/drop_front.hpp"
#include "meta/filter.hpp"
#include "meta/first.hpp"
#include "meta/flatten.hpp"
#include "meta/fold.hpp"
#include "meta/force.hpp"
#include "meta/has_type.hpp"
#include "meta/id.hpp"
#include "meta/if.hpp"
#include "meta/is_empty.hpp"
#include "meta/is_instantiation_of.hpp"
#include "meta/is_list.hpp"
#include "meta/is_meta_class.hpp"
#include "meta/is_set.hpp"
#include "meta/iseq_to_list.hpp"
#include "meta/last.hpp"
#include "meta/length.hpp"
#include "meta/list.hpp"
#include "meta/list_to_iseq.hpp"
#include "meta/logical.hpp"
#include "meta/macros.hpp"
#include "meta/make_indices.hpp"
#include "meta/mp_find.hpp"
#include "meta/not.hpp"
#include "meta/pop_back.hpp"
#include "meta/pop_front.hpp"
#include "meta/push_back.hpp"
#include "meta/push_front.hpp"
#include "meta/rename.hpp"
#include "meta/repeat.hpp"
#include "meta/replace.hpp"
#include "meta/reverse.hpp"
#include "meta/second.hpp"
#include "meta/st_contains.hpp"
#include "meta/st_position.hpp"
#include "meta/transform.hpp"
#include "meta/type_traits.hpp"
#include "meta/utility.hpp"
#include "meta/zip.hpp"
