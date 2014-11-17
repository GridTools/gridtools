{% extends "stencil.c" %}

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
    {% for arg in functor.params if arg.input -%}
    typedef const arg_type<{{ arg.id }}> {{ arg.name }};
    {% endfor %}

    //
    // the output parameters of the stencil
    //
    {% for arg in functor.params if arg.output -%}
    typedef arg_type<{{ arg.id }}> {{ arg.name }};
    {% endfor %}

    //
    // the complete list of arguments of this functor
    //
    typedef boost::mpl::vector<{%- for arg in functor.params -%}
                                  {{ arg.name }}
                                  {%- if not loop.last -%}
                                    ,
                                  {%- endif -%}
                               {%- endfor -%}> arg_list;

    //
    // the operation of the functor
    //
    template <typename Domain>
    GT_FUNCTION
    static void Do(Domain const & dom, x_interval) 
    {
        {{ functor.body.cpp }}
    }
};

{% endblock %}
