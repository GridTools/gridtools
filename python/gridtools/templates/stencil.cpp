#include "{{ stencil.hdr_file }}"

{% set numpy_arrs = functor.params.values ( )|sort(attribute='id') %}

extern "C"
{
    int run (int dim1, int dim2, int dim3, 
            {%- for p in numpy_arrs -%}
                void *{{ p.name }}
                {%- if not loop.last -%}
                , 
                {%- endif -%}
            {%- endfor -%})
    {
        return !{{ stencil.name|lower }}::test (dim1, dim2, dim3,
                                                {%- for p in numpy_arrs -%}
                                                    {{ p.name }}
                                                    {%- if not loop.last -%}
                                                    , 
                                                    {%- endif -%}
                                                {%- endfor -%});
    }
}

