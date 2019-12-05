from pygments.lexer import RegexLexer, inherit, words
from pygments.token import *
from pygments.lexers.c_cpp import CppLexer

gridtools_keywords = ((
        'accessor',
        'in_accessor',
        'inout_accessor',
        'aggregator_type',
        'arg',
        'tmp_arg',
        'param_list',
        'backward',
        'cache',
        'cells',
        'data_store',
        'dimension',
        'edges',
        'execute',
        'extent',
        'fill',
        'flush',
        'forward',
        'global_parameter',
        'grid',
        'icosahedral_topology',
        'interval',
        'intent',
        'layout_map',
        'level',
        'parallel',
        'storage_traits',
        'vertices',
        'direction',
        'sign',
        'halo_descriptor',
        'direction',
        'sign',
        'field_on_the_fly',
        'call',
        'call_proc',
        'with',
        'at'
))

gridtools_namespace = ((
	'cache_io_policy',
	'cache_type',
))

gridtools_functions = ((
	'define_caches',
	'make_computation',
	'make_expandable_computation',
	'make_global_parameter',
	'update_global_parameter',
	'make_host_view',
	'make_target_view',
	'make_multistage',
	'make_stage',
    'boundary',
    'halo_exchange_dynamic_ut',
    'halo_exchange_generic',
))

gridtools_macros = ((
	'GT_FUNCTION',
))

class GridToolsLexer(CppLexer):
	name = "gridtools"
	aliases = ['gridtools']

	tokens = {
		'statement': [
			(words(gridtools_keywords, suffix=r'\b'), Keyword),
			(words(gridtools_functions, suffix=r'\b'), Name.Label),
			(words(gridtools_namespace, suffix=r'\b'), Name.Namespace),
			(words(gridtools_macros, suffix=r'\b'), Comment.Preproc),
			inherit,
		]
	}
