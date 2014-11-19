{% extends "stencil.h" %}

{% block functor %}

//
// definition of the operators that compose the multistage stencil:
// this is extracted from the AST analysis of the loop operations
// in Python, using the 'kernel' function as a starting point
//
struct {{ functor.name }}
{
    //
    // the number of arguments the function receives in Python,
    // excluding the 'self' parameter
    //
    static const int n_args = {{ functor.params|length }};

    //
    // the input parameters of the stencil should be 'const'
    //
    {% for name, arg in functor.params.items ( ) if arg.input -%}
    typedef const arg_type<{{ arg.id }}> {{ name }};
    {% endfor %}

    //
    // the output parameters of the stencil
    //
    {% for name, arg in functor.params.items ( ) if arg.output -%}
    typedef arg_type<{{ arg.id }}> {{ name }};
    {% endfor %}

    //
    // the complete list of arguments of this functor
    //
    typedef boost::mpl::vector<{{ functor.params.values ( )|sort(attribute='id')|join(',', attribute='name') }}> arg_list;

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
