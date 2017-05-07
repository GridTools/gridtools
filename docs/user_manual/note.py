#!/usr/bin/env python

"""
Pandoc filter to process code blocks with class "include" and
replace their content with the included file
Additional features: Pradeep Gowda <@btbytes> 2014-08-28
- include only code between "start=x" and "end=y" line numberLines
- if '.numberLines' is specified in code fence, the line numbers shown
will correspond to the actual file line numbers.
Additional annotation to the fenced code blocks:
~~~~{.python .numberLines include="hello.py" start="1" end="3"}
~~~~
Typical Usage:
pandoc sample.md -t json | ./include.py | pandoc -f json -t html -s
"""

from pandocfilters import toJSONFilter, CodeBlock, Table, Str, Header, Link, Space, split_string, AlignLeft, AlignDefault, Plain, Image


def code_include(key, value, format, meta):
    if key == 'CodeBlock':
        [[ident, classes, namevals], code] = value

        if classes[0] == "note" :
            return Table([],
                         [AlignLeft(),AlignDefault()],
                         [0,0],
                         [[],[]],
                         [
                             [[Plain([Image(["", [], [["width", "20px"], ["height", "20px"]]], [{"c": "Tip", "t": "Str"}], ["figures/hint.gif", ""])])], [Plain([Str(" ")])]],
                             [[], [Plain(split_string(code))]],
                         ])
        
        return CodeBlock([ident, classes, namevals], code)

if __name__ == "__main__":
    toJSONFilter(code_include)
