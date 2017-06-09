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
import commands as cm
from os import getpid, remove
import json

def substitute_note(key, value, format, meta):
    if key == 'CodeBlock':
        [[ident, classes, namevals], code] = value

        if classes != [] and classes[0] == "note" :
            lines = code.split("\n")
            indent_lines = ["                                                      " + s for s in lines]
            indent_code = "\n".join(indent_lines)

            table = "\n---------------------------------------------------   --------------------------------------------------------\n![Tip](../pandoc_tools/filters/imgs/hintsmall.gif)\n" + indent_code + "\n---------------------------------------------------   --------------------------------------------------------\n***"
            fname = 'tmp-' + str(getpid())
            f = open(fname, 'w')
            f.write(table);
            f.close();
            
            command = 'pandoc ../pandoc_tools/defines.md ' + fname + ' --from markdown --to json'

            (v, out) = cm.getstatusoutput(command)

            remove(fname)

            if v != 0:
                raise ValueError("Something went wrong")

            source = out.decode('utf-8')

            doc = json.loads(source)

            if 'meta' in doc:
                doc = doc['blocks']
            else:
                doc = doc[1]
                
            return doc
            #    Table([],
            #             [AlignLeft(),AlignDefault()],
            #             [0,0],
            #             [[],[]],
            #             [
            #                 [[Plain([Image(["", [], [["width", "20px"], ["height", "20px"]]], [{"c": "Tip", "t": "Str"}], ["figures/hint.gif", ""])])], [Plain([Str(" ")])]],
            #                 [[Plain([Str(" ")])], doc],
            #             ])
        
        return CodeBlock([ident, classes, namevals], code)

if __name__ == "__main__":
    toJSONFilter(substitute_note)
