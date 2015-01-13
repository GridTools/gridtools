{% block functor %}

//
// definition of the operators that compose the multistage stencil:
// this is extracted from the AST analysis of the loop operations
// in Python, using the 'kernel' function as a starting point
//
struct {{ functor.name }}
{
    //
    // the number of arguments of this functor 
    //
    static const int n_args = {{ all_params|length }};

    //
    // the input and output data fields of this functor
    // (input fields are always 'const')
    //
    {% for arg in functor_params|sort(attribute='id') %}
        {%- if arg.input %}
    typedef const arg_type<{{ arg.id }}> {{ arg.name }};
        {% else %}
    typedef arg_type<{{ arg.id }}> {{ arg.name }};
        {%- endif -%}
    {% endfor %}

    //
    // the temporary data fields of this functor
    //
    {% for arg in temp_params|sort(attribute='id') -%}
    //typedef arg_type<{{ arg.id }}, range<-1,1,-1,1> > {{ arg.name }};
    typedef arg_type<{{ arg.id }}> {{ arg.name }};
    {% endfor %}

    //
    // the complete list of arguments of this functor
    //
    typedef boost::mpl::vector<{{ all_params|sort(attribute='id')|
                                  join(',', attribute='name') }}> arg_list;

    //
    // the operation of this functor
    //
    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) 
    {
        {{ functor.body.cpp }}
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
