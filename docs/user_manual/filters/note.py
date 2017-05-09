#!/usr/bin/env python

"""
Pandoc filter to process code blocks of the form

```note
text
```
and replace their content with a table with images

Typical Usage:
pandoc sample.md -t json | python ./note.py | pandoc -f json -t html -s
"""

from pandocfilters import toJSONFilter, CodeBlock, Table, Str, Header, Link, Space, split_string, AlignLeft, AlignDefault, Plain, Image


def substitute_note(key, value, format, meta):
    if key == 'CodeBlock':
        [[ident, classes, namevals], code] = value

        if classes != [] and classes[0] == "note" :
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
    toJSONFilter(substitute_note)
