from pygments.lexer import RegexLexer, inherit, words
from pygments.token import *
from pygments.lexers.c_cpp import CppLexer

gridtools_keywords = ((
 'accessor',
 'aggregator_type',
 'arg',
 'arglist',
 'backward',
 'bpfill',
 'bypass',
 'cache',
 'cells',
 'data_store',
 'data_store_t',
 'dimension',
 'edges',
 'enumtype',
 'epflush',
 'execute',
 'expand_factor',
 'extent',
 'fill',
 'fill_and_flush',
 'flush',
 'forward',
 'global_accessor',
 'global_parameter',
 'grid',
 'icosahedral_topology',
 'IJ',
 'IJK',
 'in',
 'inout',
 'interval',
 'K',
 'layout_map',
 'level',
 'local',
 'local',
 'parallel',
 'storage_traits',
 'vector_accessor',
 'vertices',
))

gridtools_namespace = ((
	'cache_io_policy',
	'cache_type',
	'enumtype',
))

gridtools_functions = ((
	'define_caches',
	'make_computation',
	'make_device_view',
	'make_global_parameter',
	'make_host_view',
	'make_multistage',
	'make_stage',
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

