#pragma once

#include <boost/mpl/vector.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/fusion/include/push_back.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/fusion/include/transform.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/size.hpp>
#include <boost/fusion/include/nview.hpp>
#include <boost/fusion/include/zip_view.hpp>
#include <boost/fusion/view/filter_view.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/copy.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/type_traits/remove_pointer.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include "arg_type.h"
#include "storage.h"
#include "layout_map.h"

namespace _impl {
    struct l_get_type {
        template <typename U>
        struct apply {
            typedef typename U::storage_type* type;
        };
    };

    struct l_get_index {
        template <typename U>
        struct apply {
            typedef typename U::index_type type;
        };
    };

    struct l_get_it_type {
        template <typename U>
        struct apply {
            typedef typename U::storage_type::iterator_type type;
        };
    };

    struct moveto_functor {
        int i,j,k;
        moveto_functor(int i, int j, int k) 
            : i(i)
            , j(j)
            , k(k)
        {}

        template <typename t_zip_elem>
        void operator()(t_zip_elem const &a) const {
            boost::fusion::at<boost::mpl::int_<0> >(a) = &( (*(boost::fusion::at<boost::mpl::int_<1> >(a)))(i,j,k) );
        }
    };

    template <int DIR>
    struct increment_functor {
        template <typename t_zip_elem>
        void operator()(t_zip_elem const &a) const {
//             iterators[l] += (*(args[l])).template stride_along<DIR>();
            boost::fusion::at_c<0>(a) += (*(boost::fusion::at_c<1>(a))).template stride_along<DIR>();
            //            boost::fusion::at_c<0>(a) = &( (*(boost::fusion::at_c<0>(a)))(i,j,k) );
        }
    };
}

template <typename t_placeholders>
struct domain_type {
    typedef t_placeholders placeholders;
    BOOST_STATIC_CONSTANT(int, len = boost::mpl::size<placeholders>::type::value);

    typedef typename boost::mpl::transform<placeholders,
                                           _impl::l_get_type
                                           >::type raw_storage_list;

    typedef typename boost::mpl::transform<placeholders,
                                           _impl::l_get_it_type
                                           >::type raw_iterators_list;

    typedef typename boost::mpl::transform<placeholders,
                                           _impl::l_get_index
                                           >::type raw_index_list;
    
    typedef typename boost::mpl::fold<raw_index_list,
            boost::mpl::vector<>,
            typename boost::mpl::push_back<boost::mpl::_1, boost::mpl::at<raw_storage_list, boost::mpl::_2> >
    >::type arg_list_mpl;

    typedef typename boost::mpl::fold<raw_index_list,
            boost::mpl::vector<>,
            typename boost::mpl::push_back<boost::mpl::_1, boost::mpl::at<raw_iterators_list, boost::mpl::_2> >
    >::type iterator_list_mpl;
    

    typedef typename boost::fusion::result_of::as_vector<arg_list_mpl>::type arg_list;
    typedef typename boost::fusion::result_of::as_vector<iterator_list_mpl>::type iterator_list;
    
    arg_list args;
    iterator_list iterators;

private:
    typedef typename boost::fusion::vector<iterator_list&, arg_list&> zip_vector_type;
    zip_vector_type zip_vector;
    typedef typename boost::fusion::zip_view<zip_vector_type> zipping;
public:

    //explicit domain_type() {}

    struct printa {
        template <typename T>
        void operator()(T const& v) const {
            std::cout << std::hex << v << "  ";
        }
    };

    template <typename t_real_storage>
    explicit domain_type(t_real_storage const & real_storage)
        : args()
        , iterators()
        , zip_vector(iterators, args)
          //        , iter_args_zip(zip_vector)
    {
//         {
//             boost::fusion::for_each(args, printa());
//             std::cout << " <-  args" << std::endl;
//             boost::fusion::for_each(real_storage, printa());
//             std::cout << " <- real_storage" << std::endl;
//         }
//        arg_list_view ll(args);
//         {
//             boost::fusion::for_each(ll, printa());
//             std::cout << " <- arg_list_view ll(args)" << std::endl;
//         }
        boost::fusion::filter_view<arg_list , 
            boost::mpl::not_<is_temporary_storage<boost::mpl::_> > > fview(args);
//         {
//             boost::fusion::for_each(fview, printa());
//             std::cout << " <- fview" << std::endl;
//         }
        boost::fusion::copy(real_storage, fview);
//         {
//             boost::fusion::for_each(ll, printa());
//             std::cout << " <- arg_list_view ll(args)" << std::endl;
//         }
    }

    template <typename T>
    //    typename boost::mpl::at<arg_list, boost::mpl::int_<T::index> >::value_type&
    typename boost::remove_pointer<typename boost::fusion::result_of::value_at<arg_list, typename T::index_type>::type>::type::value_type&
    operator[](T const &) {
        return *(boost::fusion::template at<typename T::index_type>(iterators));
    }

//    template <int I>
//    typename boost::mpl::at<arg_list, boost::mpl::int_<I> >::value_type&
//    direct() const {
////        assert(iterators[I] >= args[I]->min_addr());
////        assert(iterators[I] < args[I]->max_addr());
//        return *(boost::fusion::at<boost::mpl::int_<I> >(iterators));
//    }

    template <typename I>
    typename boost::remove_pointer<typename boost::fusion::result_of::value_at<arg_list, I>::type>::type::value_type&
    direct() const {
        assert(boost::fusion::template at<I>(iterators) >= boost::fusion::template at<I>(args)->min_addr());
        assert(boost::fusion::template at<I>(iterators) < boost::fusion::template at<I>(args)->max_addr());
        //typename boost::remove_pointer<typename boost::fusion::result_of::value_at<arg_list, I>::type>::type::value_type v = *(boost::fusion::template at<I>(iterators));
        return *(boost::fusion::template at<I>(iterators));
    }

    template <int I, typename OFFS>
    typename boost::mpl::at<arg_list, boost::mpl::int_<I> >::value_type&
    direct(OFFS const & offset) const {
        typename boost::mpl::at<arg_list, boost::mpl::int_<I> >::value_type* ptr = iterators[I] + (*(args[I])).compute_offset(offset);
        assert(ptr >= args[I]->min_addr());
        assert(ptr < args[I]->max_addr());
        return *(ptr);
    }

    void move_to(int i, int j, int k) const {
        boost::fusion::for_each(zipping(zip_vector), _impl::moveto_functor(i,j,k));
//         for (int l = 0; l < len; ++l) {
//             iterators[l] = &( (*(args[l]))(i,j,k) );
//             assert(iterators[l] >= args[l]->min_addr());
//             assert(iterators[l] < args[l]->max_addr());
//         }
    }

    template <int DIR>
    void increment_along() const {
        boost::fusion::for_each(zipping(zip_vector), _impl::increment_functor<DIR>());
//         for (int l = 0; l < len; ++l) {
//             iterators[l] += (*(args[l])).template stride_along<DIR>();
//         }
    }

};

