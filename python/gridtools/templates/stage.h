{% block stage %}

struct {{ stage.name }}
{
    //
    // the number of arguments of this stage 
    //
    static const int n_args = {{ params|length }};

    //
    // the input data fields of this stage are marked as 'const'
    //
    {% for p in params -%}
    typedef {% if p.read_only -%}
                const
            {%- endif %} accessor<{{ loop.index0 }} {%- if p.access_pattern -%}
                                                        , range<{{ p.access_pattern|join(',') }}>
                                                    {%- endif %} > {{ p.name|replace('.', '_') }};
    {% endfor %}
    //
    // the ordered list of arguments of this stage
    //
    typedef boost::mpl::vector<{{ params|join(', ', attribute='name')|replace('.', '_') }}> arg_list;

    //
    // the operation of this stage
    //
    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) 
    {
        {{ stage.body.cpp_src }}
    }
};


//
// the following operator is provided for debugging purposes
//
std::ostream& operator<<(std::ostream& s, {{ stage.name }} const) 
{
    return s << "{{ stage.name }}";
}
 
{% endblock %}
