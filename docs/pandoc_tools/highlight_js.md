<link rel="stylesheet" href="../pandoc_tools/highlight.js/color-brewer.css">
<script src="../pandoc_tools/highlight.js/highlight.pack.js"></script>
<script src="../pandoc_tools/highlight.js/jquery-2.1.3.min.js"></script>

<script>
$(function() {
    $("pre > code").each(function(i, block) {
        var codeClass = $(this).parent().attr("class");
        if (codeClass == null || codeClass === "") {
            $(this).addClass("hljs");
        } else {
            var map = {
                js: "javascript"
            };
            if (map[codeClass]) {
                codeClass = map[codeClass];
            }
            $(this).addClass(codeClass);
            hljs.highlightBlock(this);
        }
    });
});
</script>

