{% block functor %}

//
// the definition of the operators that compose a multistage stencil
// is extracted from the AST analysis of the loop comprehensions
// in Python, which use the 'kernel' function as a starting point
//
struct {{ functor.name }}
{
    //
    // the number of arguments of this functor 
    //
    static const int n_args = {{ params|length }};

    //
    // the input data fields of this functor are marked as 'const'
    //
    {% for p in params -%}
    typedef {% if functor.scope.is_parameter (p.name, read_only=True) -%}
                const
            {%- endif %} arg_type<{{ loop.index0 }} {%- if p.range -%}
                                                        , range<{{ p.range|join(',') }}>
                                                    {%- endif %} > {{ p.name|replace('.', '_') }};
    {% endfor %}
    //
    // the ordered list of arguments of this functor
    //
    typedef boost::mpl::vector<{{ params|join(', ', attribute='name')|replace('.', '_') }}> arg_list;

    //
    // the operation of this functor
    //
    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) 
    {
        {{ functor.body.cpp_src }}
    }
};


//
// the following operator is provided for debugging purposes
//
std::ostream& operator<<(std::ostream& s, {{ functor.name }} const) 
{
    return s << "{{ functor.name }}";
}
 
{% endblock %}
