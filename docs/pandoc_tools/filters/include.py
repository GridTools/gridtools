#!/usr/bin/env python

"""
Pandoc filter to process code blocks to include one .md file
from a relative path. This works with code, too if "lang=the_lang"
is used.

In case of the lang, the laguage tag is substituted to `include`
while the other name-values pairs remain are passed to the new 
code-block, so that the options are propagated there. 

example: test.md:
>>>>>>>>>>>>>>>>>>>>>>>
text

```include
./included.md
```

```{.include lang=cpp propagated=pass_this_value}
./included.md
```

text
<<<<<<<<<<<<<<<<<<<<<<<<

Typical Usage:
pandoc test.md -t json | python ./include.py | pandoc -f json -t html -s
"""

from pandocfilters import toJSONFilter, CodeBlock, Table, Str, Header, Link, Space, split_string, AlignLeft, AlignDefault, Plain, Image
import commands as cm
import json

def code_include(key, value, format, meta):
    if key == 'CodeBlock':
        [[ident, classes, namevals], code] = value

        if classes != [] and classes[0] == "include" :
            tags = [l[0] for l in namevals if l[0]]
            if "lang" in tags:
                # switch to including into a codeblock
                vals = [l[1] for l in namevals if l[1]]
                lang = vals[tags.index("lang")]

                namevals.pop(tags.index("lang"))
                classes[0]="cpp"

                f = open(code, 'r')
                newcode = f.read()
                f.close()
                return CodeBlock([ident, classes, namevals], newcode)
            else:
                lines = code.split("\n")
                if len(lines) > 1:
                    raise ValueError("Only one file at a time can be included")


                (v, out) = cm.getstatusoutput('pandoc ../pandoc_tools/defines.md ' + code + ' --from markdown --to json | python ../pandoc_tools/filters/include.py | python ../pandoc_tools/filters/note.py ')
                if v != 0:
                    raise ValueError("Something went wrong: maybe included file " + code + " GTwas not found")

                source = out.decode('utf-8')

                doc = json.loads(source)

                if 'meta' in doc:
                    doc = doc['blocks']
                else:
                    doc=doc[1]

                # print "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
                # print doc
                # print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

                return doc

        return CodeBlock([ident, classes, namevals], code)

if __name__ == "__main__":
    toJSONFilter(code_include)
