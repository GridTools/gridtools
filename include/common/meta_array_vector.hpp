#pragma once

namespace gridtools {

    template <typename Mss1, typename Mss2, typename Cond>
    struct condition;

    template<typename Vector, typename ... Mss>
    struct meta_array_vector;

    template<typename Vector>
    struct meta_array_vector<Vector>{
        typedef Vector type;
    };

    template<typename Vector, typename First, typename ... Mss>
    struct meta_array_vector<Vector, First, Mss...>{
        typedef typename meta_array_vector<typename boost::mpl::push_back<Vector , First>::type, Mss ...>::type type;
    };

    template<typename Vector, typename Mss1, typename Mss2, typename Cond, typename ... Mss>
    struct meta_array_vector<Vector, condition<Mss1, Mss2, Cond>, Mss ... > {
        typedef condition<
            typename meta_array_vector<Vector
                                       , Mss1, Mss ...>::type
            , typename meta_array_vector<Vector
                                         , Mss2, Mss ...>::type
            , Cond
            > type;

    };

}//namespace gridtools
