#pragma once
namespace gridtools {
    namespace enumtype{

        enum isparallel {parallel_impl, serial} ;
        enum execution  {upward, downward, parallel} ;

        template<enumtype::isparallel T, enumtype::execution U=upward>
        struct execute_impl{
            static const enumtype::execution iteration=U;
            static const enumtype::isparallel execution=T;
        };

        template<enumtype::execution U>
        struct execute
        {
        typedef execute_impl<serial, U> type;
        };


        template<>
        struct execute<parallel>
        {
        typedef execute_impl<parallel_impl, upward> type;
        };


    }

} // namespace gridtools
