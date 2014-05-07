#pragma once

namespace gridtools {
    namespace _impl {
        struct instantiate_tmps {
            int tileI;
            int tileJ;
            int tileK;
        
            GT_FUNCTION
            instantiate_tmps(int tileI, int tileJ, int tileK)
                : tileI(tileI)
                , tileJ(tileJ)
                , tileK(tileK)
            {}
        
            // ElemType: an element in the data field place-holders list
            template <typename ElemType>
            GT_FUNCTION
            void operator()(ElemType  e) const {
#ifndef __CUDA_ARCH__
                typedef typename boost::fusion::result_of::value_at<ElemType, boost::mpl::int_<1> >::type range_type;
                // TODO: computed storage_type should decide where to heap/cuda allocate or stack allocate.
                typedef typename boost::remove_pointer<typename boost::remove_reference<typename boost::fusion::result_of::value_at<ElemType, boost::mpl::int_<0> >::type>::type>::type storage_type;

#ifndef NDEBUG
                std::cout << "Temporary: " << range_type() << " + (" 
                          << tileI << "x" 
                          << tileJ << "x" 
                          << tileK << ")"
                          << std::endl; 
#endif
#ifndef __CUDACC__
                std::string s = boost::lexical_cast<std::string>(range_type::iminus::value)+
                    boost::lexical_cast<std::string>(range_type::iplus::value)+
                    boost::lexical_cast<std::string>(range_type::jminus::value)+
                    boost::lexical_cast<std::string>(range_type::jplus::value);
#endif
                boost::fusion::at_c<0>(e) = new storage_type(-range_type::iminus::value+range_type::iplus::value+tileI,
                                                             -range_type::jminus::value+range_type::jplus::value+tileJ,
                                                             tileK,
#ifndef __CUDACC__
                                                             666,
                                                             s);
#else
                                                             666);
#endif
#endif
            }
        };

        struct delete_tmps {
            template <typename Elem>
            GT_FUNCTION
            void operator()(Elem & elem) const {
#ifndef __CUDA_ARCH__
                delete elem;
#endif
            }
        };

        struct call_h2d {
            template <typename Arg>
            GT_FUNCTION
            void operator()(Arg * arg) const {
#ifndef __CUDA_ARCH__
                arg->h2d_update();
#endif
            }
        };

        struct call_d2h {
            template <typename Arg>
            GT_FUNCTION
            void operator()(Arg * arg) const {
#ifndef __CUDA_ARCH__
                arg->d2h_update();
#endif
            }
        };

    } // namespace _impl

    template <typename Backend>
    struct heap_allocated_temps {
        /**
         * This function is to be called by intermediate representation or back-end
         * 
         * @tparam MssType The multistage stencil type as passed to the back-end
         * @tparam RangeSizes mpl::vector with the sizes of the extents of the 
         *         access for each functor listed as linear_esf in MssType
         * @tparam BackEnd This is not currently used and may be dropped in future
         * 
         * @param tileI Tile size in the first dimension as used by the back-end
         * @param tileJ Tile size in the second dimension as used by the back-end
         * @param tileK Tile size in the third dimension as used by the back-end
         */
        template <typename MssType, typename RangeSizes, typename Domain, typename Coords>
        static void prepare_temporaries(Domain & domain, Coords const& coords) {
            int tileI, tileJ, tileK;

            tileI = (Backend::BI)?
                (Backend::BI):
                (coords.i_high_bound()-coords.i_low_bound()+1);
        
            tileJ = (Backend::BJ)?
                (Backend::BJ):
                (coords.j_high_bound()-coords.j_low_bound()+1);

            tileK = coords.value_at_top()-coords.value_at_bottom();

#ifndef NDEBUG
            std::cout << "tileI " << tileI << " "
                      << "tileJ " << tileJ
                      << std::endl;
#endif

#ifndef NDEBUG
            std::cout << "Prepare ARGUMENTS" << std::endl;
#endif

            // Got to find temporary indices
            typedef typename boost::mpl::fold<typename Domain::placeholders,
                boost::mpl::vector<>,
                boost::mpl::if_<
            is_plchldr_to_temp<boost::mpl::_2>,
                boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2 >,
                boost::mpl::_1>
            >::type list_of_temporaries;
        
#ifndef NDEBUG
            std::cout << "BEGIN TMPS" << std::endl;
            for_each<list_of_temporaries>(_debug::print_index());
            std::cout << "END TMPS" << std::endl;
#endif
        
            // Compute a vector of vectors of temp indices of temporaris initialized by each functor
            typedef typename boost::mpl::fold<typename MssType::linear_esf,
                boost::mpl::vector<>,
                boost::mpl::push_back<boost::mpl::_1, _impl::get_temps_per_functor<boost::mpl::_2> >
                >::type temps_per_functor;

            typedef typename boost::mpl::transform<
            list_of_temporaries,
                _impl::associate_ranges<temps_per_functor, RangeSizes>
                >::type list_of_ranges;

#ifndef NDEBUG
            std::cout << "BEGIN TMPS/F" << std::endl;
            for_each<temps_per_functor>(_debug::print_tmps());
            std::cout << "END TMPS/F" << std::endl;

            std::cout << "BEGIN RANGES/F" << std::endl;
            for_each<list_of_ranges>(_debug::print_ranges());
            std::cout << "END RANGES/F" << std::endl;

            std::cout << "BEGIN Fs" << std::endl;
            for_each<typename MssType::linear_esf>(_debug::print_ranges());
            std::cout << "END Fs" << std::endl;
#endif
        
            typedef boost::fusion::filter_view<typename Domain::arg_list, 
                is_temporary_storage<boost::mpl::_> > tmp_view_type;
            tmp_view_type fview(domain.storage_pointers);

#ifndef NDEBUG
            std::cout << "BEGIN VIEW" << std::endl;
            boost::fusion::for_each(fview, _debug::print_view());
            std::cout << "END VIEW" << std::endl;
#endif
        
            list_of_ranges lor;
            typedef boost::fusion::vector<tmp_view_type&, list_of_ranges const&> zipper;
            zipper zzip(fview, lor);
            boost::fusion::zip_view<zipper> zip(zzip); 
            boost::fusion::for_each(zip, _impl::instantiate_tmps(tileI, tileJ, tileK));

#ifndef NDEBUG
            std::cout << "BEGIN VIEW DOPO" << std::endl;
            boost::fusion::for_each(fview, _debug::print_view_());
            std::cout << "END VIEW DOPO" << std::endl;
#endif        

            domain.is_ready = true;
        }

        /**
           This function calls d2h_update on all storages, in order to
           get the data back to the host after a computation.
        */
        template <typename Domain>
        static void finalize_computation(Domain & domain) {
            boost::fusion::for_each(domain.original_pointers, _impl::call_d2h());
            boost::fusion::copy(domain.original_pointers, domain.storage_pointers);
        }

    };
} // namespace gridtools


