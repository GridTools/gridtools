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

namespace gridtools {
    template < typename T >
    struct iterate_domain_local_domain;

    template < typename T >
    struct is_iterate_domain;

    template < typename T >
    struct is_positional_iterate_domain;

    template < typename T >
    struct is_iterate_domain_cache : boost::mpl::false_ {};

    template < typename IterateDomainImpl >
    struct iterate_domain_backend_id;

} // namespace gridtools
