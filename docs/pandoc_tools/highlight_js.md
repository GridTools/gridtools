<link rel="stylesheet" href="highlight.js/color-brewer.min.css">
<script src="highlight.js/highlight.min.js"></script>
<script src="highlight.js/jquery-2.1.3.min.js"></script>
<script src="highlight.js/cpp_gt.js"></script>

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

