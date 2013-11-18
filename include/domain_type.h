#pragma once

#include <boost/mpl/vector.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/fusion/include/push_back.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/fusion/include/transform.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/mpl/find.hpp>
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
#include <boost/mpl/distance.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/mpl/contains.hpp>

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

    template <typename T>
    struct is_plchldr_to_temp : boost::false_type
    {};

    template <int I, typename T>
    struct is_plchldr_to_temp<arg<I, temporary<T> > > : boost::true_type
    {};

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

    template <typename t_esf>
    struct is_written_temp {
        template <typename index>
        struct apply {
            typedef typename boost::mpl::if_<
                is_plchldr_to_temp<typename boost::mpl::at<typename t_esf::args, index>::type>,
                typename boost::mpl::if_<
                    boost::is_const<typename boost::mpl::at<typename t_esf::esf_function::arg_list, index>::type>,
                    typename boost::false_type,
                    typename boost::true_type
                    >::type,
                typename boost::false_type
            >::type type;
        };
    };

    template <typename t_esf>
    struct get_it {
        template <typename index>
        struct apply {
            typedef typename boost::mpl::at<typename t_esf::args, index>::type type;
        };
    };

    template <typename t_esf_f>
    struct get_temps_per_functor {
        typedef boost::mpl::range_c<int, 0, boost::mpl::size<typename t_esf_f::args>::type::value> range;
        typedef typename boost::mpl::fold<
            range,
            boost::mpl::vector<>,
            typename boost::mpl::if_<
                typename is_written_temp<t_esf_f>::template apply<boost::mpl::_2>,
                boost::mpl::push_back<
                    boost::mpl::_1, 
                    typename _impl::get_it<t_esf_f>::template apply<boost::mpl::_2> >,
                boost::mpl::_1
                >
            >::type type;
    };

    template <typename temps_per_functor, typename t_range_sizes>
    struct associate_ranges {

        template <typename t_temp>
        struct is_temp_there {
            template <typename t_temps_in_esf>
            struct apply {
                typedef typename boost::mpl::contains<
                    t_temps_in_esf,
                    t_temp >::type type;
            };
        };
        
        template <typename t_temp>
        struct apply {

            typedef typename boost::mpl::find_if<
                temps_per_functor,
                typename is_temp_there<t_temp>::template apply<boost::mpl::_> >::type iter;

            BOOST_MPL_ASSERT_MSG( ( boost::mpl::not_<typename boost::is_same<iter, typename boost::mpl::end<temps_per_functor>::type >::type >::type::value ) ,WHATTTTTTTTT_, (iter) );

            typedef typename boost::mpl::at<t_range_sizes, typename iter::pos>::type type;               
        };
    };
    
    template <typename t_back_end>
    struct instantiate_tmps {
        int tileI;
        int tileJ;
        int tileK;
        
        instantiate_tmps(int tileI, int tileJ, int tileK)
        : tileI(tileI)
        , tileJ(tileJ)
        , tileK(tileK)
        {}
        
        // elem_type: an element in the data field place-holders list
        template <typename elem_type>
        void operator()(elem_type  e) const {
            //int i = elem;
            typedef typename boost::fusion::result_of::value_at<elem_type, boost::mpl::int_<1> >::type range_type;
            typedef typename boost::remove_pointer<typename boost::remove_reference<typename boost::fusion::result_of::value_at<elem_type, boost::mpl::int_<0> >::type>::type>::type storage_type;
            std::cout << " HAHAHAHAHAHAH " << range_type() << std::endl;
            storage_type::text();
            boost::fusion::at_c<0>(e) = new storage_type(-range_type::iminus::value+range_type::iplus::value+tileI,
                    -range_type::jminus::value+range_type::jplus::value+tileJ,
                    tileK);
        }
    };

} // namespace _impl

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
    {
        typedef typename boost::fusion::filter_view<arg_list, 
            boost::mpl::not_<is_temporary_storage<boost::mpl::_> > > view_type;

        view_type fview(args);

        BOOST_MPL_ASSERT_MSG( (boost::fusion::result_of::size<view_type>::type::value == boost::mpl::size<t_real_storage>::type::value), _NUMBER_OF_ARGS_SEEMS_WRONG_, (boost::fusion::result_of::size<view_type>) );

        // {
        //     boost::fusion::for_each(fview, printa());
        //     std::cout << " <- fview" << std::endl;
        // }

        boost::fusion::copy(real_storage, fview);

        // {
        //     boost::fusion::for_each(ll, printa());
        //     std::cout << " <- arg_list_view ll(args)" << std::endl;
        // }
    }


    struct print_index {
        template <typename T>
        void operator()(T& ) const {
            std::cout << " *" << T() << ", " << T::index::value << " * " << std::endl;
        }
    };

    struct print_tmps {
        template <typename T>
        void operator()(T& ) const {
            boost::mpl::for_each<T>(print_index());
            std::cout << " ---" << std::endl;
        }
    };

    struct print_ranges {
        template <typename T>
        void operator()(T& ) const {
            std::cout << T() << std::endl;
        }
    };

    struct print_view {
        template <typename T>
        void operator()(T& t) const {
            // int a = T();
            boost::remove_pointer<T>::type::text();
        }
    };

    struct print_view_ {
        template <typename T>
        void operator()(T& t) const {
            // int a = T();
            t->info();
        }
    };

    template <typename t_mss_type, typename t_range_sizes, typename t_back_end>
    void prepare_temporaries(int tileI, int tileJ, int tileK) {
        std::cout << "Prepare ARGUMENTS" << std::endl;

        // Got to find temporary indices
        typedef typename boost::mpl::fold<placeholders,
            boost::mpl::vector<>,
            boost::mpl::if_<
                _impl::is_plchldr_to_temp<boost::mpl::_2>,
                 boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2 >,
                 boost::mpl::_1>
            >::type list_of_temporaries;

        std::cout << "BEGIN TMPS" << std::endl;
        boost::mpl::for_each<list_of_temporaries>(print_index());
        std::cout << "END TMPS" << std::endl;
//        static const int a = typename boost::mpl::template at<list_of_temporaries,boost::mpl::template int_<0> >::type();
        // Compute a vector of vectors of temp indices of temporaris initialized by each functor
        typedef typename boost::mpl::fold<typename t_mss_type::linear_esf,
            boost::mpl::vector<>,
            boost::mpl::push_back<boost::mpl::_1, typename _impl::get_temps_per_functor<boost::mpl::_2> >
            >::type temps_per_functor;

        typedef typename boost::mpl::transform<
            list_of_temporaries,
            _impl::associate_ranges<temps_per_functor, t_range_sizes>
        >::type list_of_ranges;
        
        // typedef typename boost::mpl::fold<list_of_temporaries,
        //     boost::mpl::vector<>,
        //     boost::mpl::push_back<boost::mpl::_1, 
        //     typename _impl::compute_range<typename t_mss_type::linear_esf>::template apply<boost::mpl::_2> > 
        //     >::type list_of_ranges;

        std::cout << "BEGIN TMPS/F" << std::endl;
        boost::mpl::for_each<temps_per_functor>(print_tmps());
        std::cout << "END TMPS/F" << std::endl;

        std::cout << "BEGIN RANGES/F" << std::endl;
        boost::mpl::for_each<list_of_ranges>(print_ranges());
        std::cout << "END RANGES/F" << std::endl;

        std::cout << "BEGIN Fs" << std::endl;
        boost::mpl::for_each<typename t_mss_type::linear_esf>(print_ranges());
        std::cout << "END Fs" << std::endl;

        typedef typename boost::fusion::filter_view<arg_list, 
            is_temporary_storage<boost::mpl::_> > tmp_view_type;
        tmp_view_type fview(args);

        std::cout << "BEGIN VIEW" << std::endl;
        boost::fusion::for_each(fview, print_view());
        std::cout << "END VIEW" << std::endl;
        
        list_of_ranges lor;
        typedef typename boost::fusion::vector<tmp_view_type&, list_of_ranges const&> zipper;
        zipper zzip(fview, lor);
        boost::fusion::zip_view<zipper> zip(zzip); 
        boost::fusion::for_each(zip, _impl::instantiate_tmps< t_back_end >(tileI, tileJ, tileK));

        std::cout << "BEGIN VIEW DOPO" << std::endl;
        boost::fusion::for_each(fview, print_view_());
        std::cout << "END VIEW DOPO" << std::endl;
        
    }

    template <typename T>
    typename boost::remove_pointer<typename boost::fusion::result_of::value_at<arg_list, typename T::index_type>::type>::type::value_type&
    operator[](T const &) {
        return *(boost::fusion::template at<typename T::index_type>(iterators));
    }

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

