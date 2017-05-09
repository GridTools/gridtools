#!/usr/bin/env python

"""
Pandoc filter to process code blocks to include one .md file
from a relative path

```include
filepath/name.md
```

Typical Usage:
pandoc sample.md -t json | python ./include.py | pandoc -f json -t html -s
"""

from pandocfilters import toJSONFilter, CodeBlock, Table, Str, Header, Link, Space, split_string, AlignLeft, AlignDefault, Plain, Image
import commands as cm
import json



def code_include(key, value, format, meta):
    if key == 'CodeBlock':
        [[ident, classes, namevals], code] = value

        if classes != [] and classes[0] == "include" :
            
            lines = code.split("\n")
            if len(lines) > 1:
                raise ValueError("Only one file at a time can be included")


            (v, out) = cm.getstatusoutput('pandoc ' + code + ' --from markdown --to json ')
            if v != 0:
                raise ValueError("Something went wrong: maybe included file was not found")
            source = out.decode('utf-8')
            doc = json.loads(source)
            return doc[1]

        return CodeBlock([ident, classes, namevals], code)

if __name__ == "__main__":
    toJSONFilter(code_include)
