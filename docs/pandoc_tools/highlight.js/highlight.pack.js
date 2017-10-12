/*! highlight.js v9.12.0 | BSD3 License | git.io/hljslicense */
!function(e){var t="object"==typeof window&&window||"object"==typeof self&&self;"undefined"!=typeof exports?e(exports):t&&(t.hljs=e({}),"function"==typeof define&&define.amd&&define([],function(){return t.hljs}))}(function(e){function t(e){return e.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}function n(e){return e.nodeName.toLowerCase()}function r(e,t){var n=e&&e.exec(t);return n&&0===n.index}function a(e){return C.test(e)}function i(e){var t,n,r,i,c=e.className+" ";if(c+=e.parentNode?e.parentNode.className:"",n=E.exec(c))return w(n[1])?n[1]:"no-highlight";for(c=c.split(/\s+/),t=0,r=c.length;r>t;t++)if(i=c[t],a(i)||w(i))return i}function c(e){var t,n={},r=Array.prototype.slice.call(arguments,1);for(t in e)n[t]=e[t];return r.forEach(function(e){for(t in e)n[t]=e[t]}),n}function o(e){var t=[];return function r(e,a){for(var i=e.firstChild;i;i=i.nextSibling)3===i.nodeType?a+=i.nodeValue.length:1===i.nodeType&&(t.push({event:"start",offset:a,node:i}),a=r(i,a),n(i).match(/br|hr|img|input/)||t.push({event:"stop",offset:a,node:i}));return a}(e,0),t}function s(e,r,a){function i(){return e.length&&r.length?e[0].offset!==r[0].offset?e[0].offset<r[0].offset?e:r:"start"===r[0].event?e:r:e.length?e:r}function c(e){function r(e){return" "+e.nodeName+'="'+t(e.value).replace('"',"&quot;")+'"'}l+="<"+n(e)+x.map.call(e.attributes,r).join("")+">"}function o(e){l+="</"+n(e)+">"}function s(e){("start"===e.event?c:o)(e.node)}for(var u=0,l="",f=[];e.length||r.length;){var p=i();if(l+=t(a.substring(u,p[0].offset)),u=p[0].offset,p===e){f.reverse().forEach(o);do s(p.splice(0,1)[0]),p=i();while(p===e&&p.length&&p[0].offset===u);f.reverse().forEach(c)}else"start"===p[0].event?f.push(p[0].node):f.pop(),s(p.splice(0,1)[0])}return l+t(a.substr(u))}function u(e){return e.v&&!e.cached_variants&&(e.cached_variants=e.v.map(function(t){return c(e,{v:null},t)})),e.cached_variants||e.eW&&[c(e)]||[e]}function l(e){function t(e){return e&&e.source||e}function n(n,r){return new RegExp(t(n),"m"+(e.cI?"i":"")+(r?"g":""))}function r(a,i){if(!a.compiled){if(a.compiled=!0,a.k=a.k||a.bK,a.k){var c={},o=function(t,n){e.cI&&(n=n.toLowerCase()),n.split(" ").forEach(function(e){var n=e.split("|");c[n[0]]=[t,n[1]?Number(n[1]):1]})};"string"==typeof a.k?o("keyword",a.k):k(a.k).forEach(function(e){o(e,a.k[e])}),a.k=c}a.lR=n(a.l||/\w+/,!0),i&&(a.bK&&(a.b="\\b("+a.bK.split(" ").join("|")+")\\b"),a.b||(a.b=/\B|\b/),a.bR=n(a.b),a.e||a.eW||(a.e=/\B|\b/),a.e&&(a.eR=n(a.e)),a.tE=t(a.e)||"",a.eW&&i.tE&&(a.tE+=(a.e?"|":"")+i.tE)),a.i&&(a.iR=n(a.i)),null==a.r&&(a.r=1),a.c||(a.c=[]),a.c=Array.prototype.concat.apply([],a.c.map(function(e){return u("self"===e?a:e)})),a.c.forEach(function(e){r(e,a)}),a.starts&&r(a.starts,i);var s=a.c.map(function(e){return e.bK?"\\.?("+e.b+")\\.?":e.b}).concat([a.tE,a.i]).map(t).filter(Boolean);a.t=s.length?n(s.join("|"),!0):{exec:function(){return null}}}}r(e)}function f(e,n,a,i){function c(e,t){var n,a;for(n=0,a=t.c.length;a>n;n++)if(r(t.c[n].bR,e))return t.c[n]}function o(e,t){if(r(e.eR,t)){for(;e.endsParent&&e.parent;)e=e.parent;return e}return e.eW?o(e.parent,t):void 0}function s(e,t){return!a&&r(t.iR,e)}function u(e,t){var n=_.cI?t[0].toLowerCase():t[0];return e.k.hasOwnProperty(n)&&e.k[n]}function d(e,t,n,r){var a=r?"":M.classPrefix,i='<span class="'+a,c=n?"":B;return i+=e+'">',i+t+c}function g(){var e,n,r,a;if(!x.k)return t(C);for(a="",n=0,x.lR.lastIndex=0,r=x.lR.exec(C);r;)a+=t(C.substring(n,r.index)),e=u(x,r),e?(E+=e[1],a+=d(e[0],t(r[0]))):a+=t(r[0]),n=x.lR.lastIndex,r=x.lR.exec(C);return a+t(C.substr(n))}function m(){var e="string"==typeof x.sL;if(e&&!y[x.sL])return t(C);var n=e?f(x.sL,C,!0,k[x.sL]):p(C,x.sL.length?x.sL:void 0);return x.r>0&&(E+=n.r),e&&(k[x.sL]=n.top),d(n.language,n.value,!1,!0)}function b(){R+=null!=x.sL?m():g(),C=""}function h(e){R+=e.cN?d(e.cN,"",!0):"",x=Object.create(e,{parent:{value:x}})}function v(e,t){if(C+=e,null==t)return b(),0;var n=c(t,x);if(n)return n.skip?C+=t:(n.eB&&(C+=t),b(),n.rB||n.eB||(C=t)),h(n,t),n.rB?0:t.length;var r=o(x,t);if(r){var a=x;a.skip?C+=t:(a.rE||a.eE||(C+=t),b(),a.eE&&(C=t));do x.cN&&(R+=B),x.skip||(E+=x.r),x=x.parent;while(x!==r.parent);return r.starts&&h(r.starts,""),a.rE?0:t.length}if(s(t,x))throw new Error('Illegal lexeme "'+t+'" for mode "'+(x.cN||"<unnamed>")+'"');return C+=t,t.length||1}var _=w(e);if(!_)throw new Error('Unknown language: "'+e+'"');l(_);var N,x=i||_,k={},R="";for(N=x;N!==_;N=N.parent)N.cN&&(R=d(N.cN,"",!0)+R);var C="",E=0;try{for(var L,I,j=0;;){if(x.t.lastIndex=j,L=x.t.exec(n),!L)break;I=v(n.substring(j,L.index),L[0]),j=L.index+I}for(v(n.substr(j)),N=x;N.parent;N=N.parent)N.cN&&(R+=B);return{r:E,value:R,language:e,top:x}}catch(T){if(T.message&&-1!==T.message.indexOf("Illegal"))return{r:0,value:t(n)};throw T}}function p(e,n){n=n||M.languages||k(y);var r={r:0,value:t(e)},a=r;return n.filter(w).forEach(function(t){var n=f(t,e,!1);n.language=t,n.r>a.r&&(a=n),n.r>r.r&&(a=r,r=n)}),a.language&&(r.second_best=a),r}function d(e){return M.tabReplace||M.useBR?e.replace(L,function(e,t){return M.useBR&&"\n"===e?"<br>":M.tabReplace?t.replace(/\t/g,M.tabReplace):""}):e}function g(e,t,n){var r=t?R[t]:n,a=[e.trim()];return e.match(/\bhljs\b/)||a.push("hljs"),-1===e.indexOf(r)&&a.push(r),a.join(" ").trim()}function m(e){var t,n,r,c,u,l=i(e);a(l)||(M.useBR?(t=document.createElementNS("http://www.w3.org/1999/xhtml","div"),t.innerHTML=e.innerHTML.replace(/\n/g,"").replace(/<br[ \/]*>/g,"\n")):t=e,u=t.textContent,r=l?f(l,u,!0):p(u),n=o(t),n.length&&(c=document.createElementNS("http://www.w3.org/1999/xhtml","div"),c.innerHTML=r.value,r.value=s(n,o(c),u)),r.value=d(r.value),e.innerHTML=r.value,e.className=g(e.className,l,r.language),e.result={language:r.language,re:r.r},r.second_best&&(e.second_best={language:r.second_best.language,re:r.second_best.r}))}function b(e){M=c(M,e)}function h(){if(!h.called){h.called=!0;var e=document.querySelectorAll("pre code");x.forEach.call(e,m)}}function v(){addEventListener("DOMContentLoaded",h,!1),addEventListener("load",h,!1)}function _(t,n){var r=y[t]=n(e);r.aliases&&r.aliases.forEach(function(e){R[e]=t})}function N(){return k(y)}function w(e){return e=(e||"").toLowerCase(),y[e]||y[R[e]]}var x=[],k=Object.keys,y={},R={},C=/^(no-?highlight|plain|text)$/i,E=/\blang(?:uage)?-([\w-]+)\b/i,L=/((^(<[^>]+>|\t|)+|(?:\n)))/gm,B="</span>",M={classPrefix:"hljs-",tabReplace:null,useBR:!1,languages:void 0};return e.highlight=f,e.highlightAuto=p,e.fixMarkup=d,e.highlightBlock=m,e.configure=b,e.initHighlighting=h,e.initHighlightingOnLoad=v,e.registerLanguage=_,e.listLanguages=N,e.getLanguage=w,e.inherit=c,e.IR="[a-zA-Z]\\w*",e.UIR="[a-zA-Z_]\\w*",e.NR="\\b\\d+(\\.\\d+)?",e.CNR="(-?)(\\b0[xX][a-fA-F0-9]+|(\\b\\d+(\\.\\d*)?|\\.\\d+)([eE][-+]?\\d+)?)",e.BNR="\\b(0b[01]+)",e.RSR="!|!=|!==|%|%=|&|&&|&=|\\*|\\*=|\\+|\\+=|,|-|-=|/=|/|:|;|<<|<<=|<=|<|===|==|=|>>>=|>>=|>=|>>>|>>|>|\\?|\\[|\\{|\\(|\\^|\\^=|\\||\\|=|\\|\\||~",e.BE={b:"\\\\[\\s\\S]",r:0},e.ASM={cN:"string",b:"'",e:"'",i:"\\n",c:[e.BE]},e.QSM={cN:"string",b:'"',e:'"',i:"\\n",c:[e.BE]},e.PWM={b:/\b(a|an|the|are|I'm|isn't|don't|doesn't|won't|but|just|should|pretty|simply|enough|gonna|going|wtf|so|such|will|you|your|they|like|more)\b/},e.C=function(t,n,r){var a=e.inherit({cN:"comment",b:t,e:n,c:[]},r||{});return a.c.push(e.PWM),a.c.push({cN:"doctag",b:"(?:TODO|FIXME|NOTE|BUG|XXX):",r:0}),a},e.CLCM=e.C("//","$"),e.CBCM=e.C("/\\*","\\*/"),e.HCM=e.C("#","$"),e.NM={cN:"number",b:e.NR,r:0},e.CNM={cN:"number",b:e.CNR,r:0},e.BNM={cN:"number",b:e.BNR,r:0},e.CSSNM={cN:"number",b:e.NR+"(%|em|ex|ch|rem|vw|vh|vmin|vmax|cm|mm|in|pt|pc|px|deg|grad|rad|turn|s|ms|Hz|kHz|dpi|dpcm|dppx)?",r:0},e.RM={cN:"regexp",b:/\//,e:/\/[gimuy]*/,i:/\n/,c:[e.BE,{b:/\[/,e:/\]/,r:0,c:[e.BE]}]},e.TM={cN:"title",b:e.IR,r:0},e.UTM={cN:"title",b:e.UIR,r:0},e.METHOD_GUARD={b:"\\.\\s*"+e.UIR,r:0},e.registerLanguage("cpp-gt",function(e){var t={cN:"keyword",b:"\\b[a-z\\d_]*_t\\b"},n={cN:"string",v:[{b:'(u8?|U)?L?"',e:'"',i:"\\n",c:[e.BE]},{b:'(u8?|U)?R"',e:'"',c:[e.BE]},{b:"'\\\\?.",e:"'",i:"."}]},r={cN:"number",v:[{b:"\\b(0b[01']+)"},{b:"(-?)\\b([\\d']+(\\.[\\d']*)?|\\.[\\d']+)(u|U|l|L|ul|UL|f|F|b|B)"},{b:"(-?)(\\b0[xX][a-fA-F0-9']+|(\\b[\\d']+(\\.[\\d']*)?|\\.[\\d']+)([eE][-+]?[\\d']+)?)"}],r:0},a={cN:"meta",b:/#\s*[a-z]+\b/,e:/$/,k:{"meta-keyword":"if else elif endif define undef warning error line pragma ifdef ifndef include"},c:[{b:/\\\n/,r:0},e.inherit(n,{cN:"meta-string"}),{cN:"meta-string",b:/<[^\n>]*>/,e:/$/,i:"\\n"},e.CLCM,e.CBCM]},i=e.IR+"\\s*\\(",c={keyword:"int float while private char catch import module export virtual operator sizeof dynamic_cast|10 typedef const_cast|10 const for static_cast|10 union namespace unsigned long volatile static protected bool template mutable if public friend do goto auto void enum else break extern using asm case typeid short reinterpret_cast|10 default double register explicit signed typename try this switch continue inline delete alignof constexpr decltype noexcept static_assert thread_local restrict _Bool complex _Complex _Imaginary atomic_bool atomic_char atomic_schar atomic_uchar atomic_short atomic_ushort atomic_int atomic_uint atomic_long atomic_ulong atomic_llong atomic_ullong new throw return and or not",built_in:"std string cin cout cerr clog stdin stdout stderr stringstream istringstream ostringstream auto_ptr deque list queue stack vector map set bitset multiset multimap unordered_set unordered_map unordered_multiset unordered_multimap array shared_ptr abort abs acos asin atan2 atan calloc ceil cosh cos exit exp fabs floor fmod fprintf fputs free frexp fscanf isalnum isalpha iscntrl isdigit isgraph islower isprint ispunct isspace isupper isxdigit tolower toupper labs ldexp log10 log malloc realloc memchr memcmp memcpy memset modf pow printf putchar puts scanf sinh sin snprintf sprintf sqrt sscanf strcat strchr strcmp strcpy strcspn strlen strncat strncmp strncpy strpbrk strrchr strspn strstr tanh tan vfprintf vprintf vsprintf endl initializer_list unique_ptr",literal:"true false nullptr NULL",meta:"halo_descriptor interval level layout_map Cuda Host Block Naive structured icosahedral dimension accessor global_accessor in_accessor inout_accessor extent make_stage make_computation make_multistage make_independent forward define_caches execute cache fill_and_flush fill flush epflush bpfill local IJ K IJK bypass"},o=[t,e.CLCM,e.CBCM,r,n];return{aliases:["c","cc","h","c++","h++","hpp"],k:c,i:"</",c:o.concat([a,{b:"\\b(deque|list|queue|stack|vector|map|set|bitset|multiset|multimap|unordered_map|unordered_set|unordered_multiset|unordered_multimap|array)\\s*<",e:">",k:c,c:["self",t]},{b:e.IR+"::",k:c},{v:[{b:/=/,e:/;/},{b:/\(/,e:/\)/},{bK:"new throw return else",e:/;/}],k:c,c:o.concat([{b:/\(/,e:/\)/,k:c,c:o.concat(["self"]),r:0}]),r:0},{cN:"function",b:"("+e.IR+"[\\*&\\s]+)+"+i,rB:!0,e:/[{;=]/,eE:!0,k:c,i:/[^\w\s\*&]/,c:[{b:i,rB:!0,c:[e.TM],r:0},{cN:"params",b:/\(/,e:/\)/,k:c,r:0,c:[e.CLCM,e.CBCM,n,r,t]},e.CLCM,e.CBCM,a]},{cN:"class",bK:"class struct",e:/[{;:]/,c:[{b:/</,e:/>/,c:["self"]},e.TM]}]),exports:{preprocessor:a,strings:n,k:c}}}),e});
