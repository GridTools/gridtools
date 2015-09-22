;; load this file in your emacs configuration file, by adding the following line at the beginning of your .emacs
;; (load-file "<path>/.gridtools.el")
;; where <path> is the location path for this file

(setq c-default-style '((java-mode . "java") (awk-mode . "awk") (other .  "bsd" ) ))
(setq-default c-basic-offset 4)
(setq-default indicate-empty-lines t)
(setq-default show-trailing-whitespace t)
(add-hook 'before-save-hook 'delete-trailing-whitespace)
