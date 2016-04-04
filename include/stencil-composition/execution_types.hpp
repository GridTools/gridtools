#pragma once
namespace gridtools {
    namespace enumtype {

        enum isparallel { parallel_impl, serial };
        enum execution { forward, backward, parallel };

        template < enumtype::isparallel T, enumtype::execution U = forward >
        struct execute_impl {
            static const enumtype::execution iteration = U;
            static const enumtype::isparallel execution = T;
        };

        template < enumtype::execution U >
        struct execute {
            typedef execute_impl< serial, U > type;
        };

        template <>
        struct execute< parallel > {
            typedef execute_impl< parallel_impl, forward > type;
        };
    }

    template < typename T >
    struct is_execution_engine : boost::mpl::false_ {};

    template < enumtype::execution U >
    struct is_execution_engine< enumtype::execute< U > > : boost::mpl::true_ {};

} // namespace gridtools
