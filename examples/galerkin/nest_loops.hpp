#pragma once
//TODO (optional) : rewrite generic
namespace gridtools{

    template<typename ArrayKnots, typename Quad, typename Storage
	      , template<typename AK, typename Q, typename S, int ... I> class InnerFunctor
	     , typename ... Ranges>
    struct nest_loop;

    template<typename ArrayKnots, typename Quad, typename Storage
	      , template<typename AK, typename Q, typename S, int ... I> class InnerFunctor
	     , typename Range1>
    struct nest_loop<ArrayKnots, Quad, Storage, InnerFunctor, Range1>{

	using array_t=ArrayKnots;

	Quad const& m_quad;
	Storage& m_storage;
	array_t const& m_knots;

	nest_loop(Quad const& quad_points_, Storage& storage_, array_t const& knots_)
	    :
	    m_quad(quad_points_)
	    , m_storage(storage_)
	    , m_knots(knots_)
	{}

	template <typename Id>
	void operator()(Id ){
	    boost::mpl::for_each<Range1>(InnerFunctor<ArrayKnots, Quad, Storage, Id::value>(m_quad, m_storage, m_knots));
        }

	void operator()(){
	    boost::mpl::for_each<Range1>(InnerFunctor<ArrayKnots, Quad, Storage>(m_quad, m_storage, m_knots));
	}

    };


    template <typename ArrayKnots, typename Quad, typename Storage
	      , template<typename AK, typename Q, typename S, int ... I> class InnerFunctor
	      , typename Range1, typename Range2, typename Range3>
    struct nest_loop<ArrayKnots, Quad, Storage, InnerFunctor, Range1, Range2, Range3>{

	using array_t=ArrayKnots;

	Quad const& m_quad;
	Storage& m_storage;
	array_t const& m_knots;

	nest_loop(Quad const& quad_points_, Storage& storage_, array_t const& knots_)
	    :
	    m_quad(quad_points_)
	    , m_storage(storage_)
	    , m_knots(knots_)
	{}

	void operator()(){
	    boost::mpl::for_each<Range1>(nest_loop<ArrayKnots, Quad, Storage, InnerFunctor, Range2, Range3>(m_quad, m_storage, m_knots));
	}

	template <typename Id>
	void operator()(Id ){
	    boost::mpl::for_each<Range2>(nest_loop<ArrayKnots, Quad, Storage, InnerFunctor, Id, Range2, Range3>(m_quad, m_storage, m_knots));
	}

    };


    template <typename ArrayKnots, typename Quad, typename Storage, template<typename AK, typename Q, typename S, int ... I> class InnerFunctor, typename Range2, typename Range3>
    struct nest_loop<ArrayKnots, Quad, Storage, InnerFunctor, Range2, Range3>{

	using array_t=ArrayKnots;

	Quad const& m_quad;
	Storage& m_storage;
	array_t const& m_knots;

	nest_loop(Quad const& quad_points_, Storage& storage_, array_t const& knots_)
	    :
	    m_quad(quad_points_)
	    , m_storage(storage_)
	    , m_knots(knots_)
	{}

	void operator()(){
	    boost::mpl::for_each<Range2>(nest_loop<ArrayKnots, Quad, Storage, InnerFunctor, Range3>(m_quad, m_storage, m_knots));
	}


	template <typename Id>
	void operator()(Id ){
	    boost::mpl::for_each<Range2>(nest_loop<ArrayKnots, Quad, Storage, InnerFunctor, Id, Range3>(m_quad, m_storage, m_knots));
	}

    };


    template <typename ArrayKnots, typename Quad, typename Storage, template<typename AK, typename Q, typename S, int ... I> class InnerFunctor, int I, typename Range3>
    struct nest_loop<ArrayKnots, Quad, Storage, InnerFunctor, static_int<I>,  Range3>{

	using array_t=ArrayKnots;

	Quad const& m_quad;
	Storage& m_storage;
	array_t const& m_knots;

	nest_loop(Quad const& quad_points_, Storage& storage_, array_t const& knots_)
	    :
	    m_quad(quad_points_)
	    , m_storage(storage_)
	    , m_knots(knots_)
	{}

	template <typename Id>
	void operator()(Id){
	    boost::mpl::for_each<Range3>(InnerFunctor<ArrayKnots, Quad, Storage, I, Id::value>(m_quad, m_storage, m_knots));
	}
    };

} //namespace gridtools
