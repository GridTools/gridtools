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
            {%- endif -%}
            arg_type<{{ loop.index0 }}> {{ p.name }};
    {% endfor %}

    //
    // the ordered list of arguments of this functor
    //
    typedef boost::mpl::vector<{{ params|join(',', attribute='name') }}> arg_list;

    //
    // the operation of this functor
    //
    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) 
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
