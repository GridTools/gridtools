#include "{{ stencil.hdr_file }}"



extern "C"
{
    int run (uint_t dim1, uint_t dim2, uint_t dim3, 
            {## 
             ## Pointers to NumPy arrays passed from Python 
             ##}
            {%- for name,arg in stencil.symbols.items ( ) if stencil.symbols.is_parameter (name) or
                                                             stencil.symbols.is_temporary (name) -%}
                void * {{ name }}_buff
                {%- if not loop.last -%}
                    ,
                {%- endif -%}
            {%- endfor -%})
    {
        return !{{ stencil.name|lower }}::test (dim1, dim2, dim3,
                                                {## 
                                                 ## Pointers to NumPy arrays passed from Python 
                                                 ##}
                                                {%- for name,arg in stencil.symbols.items ( ) if stencil.symbols.is_parameter (name) or
                                                                                                 stencil.symbols.is_temporary (name) -%}
                                                    void * {{ name }}_buff
                                                    {%- if not loop.last -%}
                                                        ,
                                                    {%- endif -%}
                                                {%- endfor -%});
    }
}

