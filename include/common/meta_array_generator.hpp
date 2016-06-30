/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once
/**
@file implementation of a metafunction to generate a vector, which is then to be used to construct a \ref
gridtools::meta_array
*/
namespace gridtools {

    template < typename Mss1, typename Mss2, typename Cond >
    struct condition;

    template < typename Vector, typename... Mss >
    struct meta_array_generator;

    /**@brief recursion anchor*/
    template < typename Vector >
    struct meta_array_generator< Vector > {
        typedef Vector type;
    };

    /**
       @brief metafunction to construct a vector of multi-stage stencils and conditions
     */
    template < typename Vector, typename First, typename... Mss >
    struct meta_array_generator< Vector, First, Mss... > {
        typedef
            typename meta_array_generator< typename boost::mpl::push_back< Vector, First >::type, Mss... >::type type;
    };

    /**
       @brief metafunction to construct a vector of multi-stage stencils and conditions

       specialization for conditions.
     */
    template < typename Vector, typename Mss1, typename Mss2, typename Cond, typename... Mss >
    struct meta_array_generator< Vector, condition< Mss1, Mss2, Cond >, Mss... > {
        typedef condition< typename meta_array_generator< Vector, Mss1, Mss... >::type,
            typename meta_array_generator< Vector, Mss2, Mss... >::type,
            Cond > type;
    };

} // namespace gridtools
