# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Paul Mensonides 2002.
#  *     Distributed under the Boost Software License, Version 1.0. (See
#  *     accompanying file LICENSE_1_0.txt or copy at
#  *     http://www.boost.org/LICENSE_1_0.txt)
#  *                                                                          *
#  ************************************************************************** */
#
# /* Revised by Edward Diener (2020) */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_SEQ_ELEM_HPP
# define GT_PREPROCESSOR_SEQ_ELEM_HPP
#
# include <gridtools/preprocessor/cat.hpp>
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/facilities/empty.hpp>
#
# /* GT_PP_SEQ_ELEM */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC()
#    define GT_PP_SEQ_ELEM(i, seq) GT_PP_SEQ_ELEM_I(i, seq)
# else
#    define GT_PP_SEQ_ELEM(i, seq) GT_PP_SEQ_ELEM_I((i, seq))
# endif
#
# if GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MSVC()
#    define GT_PP_SEQ_ELEM_I(i, seq) GT_PP_SEQ_ELEM_II((GT_PP_SEQ_ELEM_ ## i seq))
#    define GT_PP_SEQ_ELEM_II(res) GT_PP_SEQ_ELEM_IV(GT_PP_SEQ_ELEM_III res)
#    define GT_PP_SEQ_ELEM_III(x, _) x GT_PP_EMPTY()
#    define GT_PP_SEQ_ELEM_IV(x) x
# elif GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC()
#    define GT_PP_SEQ_ELEM_I(par) GT_PP_SEQ_ELEM_II ## par
#    define GT_PP_SEQ_ELEM_II(i, seq) GT_PP_SEQ_ELEM_III(GT_PP_SEQ_ELEM_ ## i ## seq)
#    define GT_PP_SEQ_ELEM_III(im) GT_PP_SEQ_ELEM_IV(im)
#    define GT_PP_SEQ_ELEM_IV(x, _) x
# else
#    if defined(__IBMC__) || defined(__IBMCPP__)
#        define GT_PP_SEQ_ELEM_I(i, seq) GT_PP_SEQ_ELEM_II(GT_PP_CAT(GT_PP_SEQ_ELEM_ ## i, seq))
#    else
#        define GT_PP_SEQ_ELEM_I(i, seq) GT_PP_SEQ_ELEM_II(GT_PP_SEQ_ELEM_ ## i seq)
#    endif
#    define GT_PP_SEQ_ELEM_II(im) GT_PP_SEQ_ELEM_III(im)
#    define GT_PP_SEQ_ELEM_III(x, _) x
# endif
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_STRICT()
#
# define GT_PP_SEQ_ELEM_0(x) x, GT_PP_NIL
# define GT_PP_SEQ_ELEM_1(_) GT_PP_SEQ_ELEM_0
# define GT_PP_SEQ_ELEM_2(_) GT_PP_SEQ_ELEM_1
# define GT_PP_SEQ_ELEM_3(_) GT_PP_SEQ_ELEM_2
# define GT_PP_SEQ_ELEM_4(_) GT_PP_SEQ_ELEM_3
# define GT_PP_SEQ_ELEM_5(_) GT_PP_SEQ_ELEM_4
# define GT_PP_SEQ_ELEM_6(_) GT_PP_SEQ_ELEM_5
# define GT_PP_SEQ_ELEM_7(_) GT_PP_SEQ_ELEM_6
# define GT_PP_SEQ_ELEM_8(_) GT_PP_SEQ_ELEM_7
# define GT_PP_SEQ_ELEM_9(_) GT_PP_SEQ_ELEM_8
# define GT_PP_SEQ_ELEM_10(_) GT_PP_SEQ_ELEM_9
# define GT_PP_SEQ_ELEM_11(_) GT_PP_SEQ_ELEM_10
# define GT_PP_SEQ_ELEM_12(_) GT_PP_SEQ_ELEM_11
# define GT_PP_SEQ_ELEM_13(_) GT_PP_SEQ_ELEM_12
# define GT_PP_SEQ_ELEM_14(_) GT_PP_SEQ_ELEM_13
# define GT_PP_SEQ_ELEM_15(_) GT_PP_SEQ_ELEM_14
# define GT_PP_SEQ_ELEM_16(_) GT_PP_SEQ_ELEM_15
# define GT_PP_SEQ_ELEM_17(_) GT_PP_SEQ_ELEM_16
# define GT_PP_SEQ_ELEM_18(_) GT_PP_SEQ_ELEM_17
# define GT_PP_SEQ_ELEM_19(_) GT_PP_SEQ_ELEM_18
# define GT_PP_SEQ_ELEM_20(_) GT_PP_SEQ_ELEM_19
# define GT_PP_SEQ_ELEM_21(_) GT_PP_SEQ_ELEM_20
# define GT_PP_SEQ_ELEM_22(_) GT_PP_SEQ_ELEM_21
# define GT_PP_SEQ_ELEM_23(_) GT_PP_SEQ_ELEM_22
# define GT_PP_SEQ_ELEM_24(_) GT_PP_SEQ_ELEM_23
# define GT_PP_SEQ_ELEM_25(_) GT_PP_SEQ_ELEM_24
# define GT_PP_SEQ_ELEM_26(_) GT_PP_SEQ_ELEM_25
# define GT_PP_SEQ_ELEM_27(_) GT_PP_SEQ_ELEM_26
# define GT_PP_SEQ_ELEM_28(_) GT_PP_SEQ_ELEM_27
# define GT_PP_SEQ_ELEM_29(_) GT_PP_SEQ_ELEM_28
# define GT_PP_SEQ_ELEM_30(_) GT_PP_SEQ_ELEM_29
# define GT_PP_SEQ_ELEM_31(_) GT_PP_SEQ_ELEM_30
# define GT_PP_SEQ_ELEM_32(_) GT_PP_SEQ_ELEM_31
# define GT_PP_SEQ_ELEM_33(_) GT_PP_SEQ_ELEM_32
# define GT_PP_SEQ_ELEM_34(_) GT_PP_SEQ_ELEM_33
# define GT_PP_SEQ_ELEM_35(_) GT_PP_SEQ_ELEM_34
# define GT_PP_SEQ_ELEM_36(_) GT_PP_SEQ_ELEM_35
# define GT_PP_SEQ_ELEM_37(_) GT_PP_SEQ_ELEM_36
# define GT_PP_SEQ_ELEM_38(_) GT_PP_SEQ_ELEM_37
# define GT_PP_SEQ_ELEM_39(_) GT_PP_SEQ_ELEM_38
# define GT_PP_SEQ_ELEM_40(_) GT_PP_SEQ_ELEM_39
# define GT_PP_SEQ_ELEM_41(_) GT_PP_SEQ_ELEM_40
# define GT_PP_SEQ_ELEM_42(_) GT_PP_SEQ_ELEM_41
# define GT_PP_SEQ_ELEM_43(_) GT_PP_SEQ_ELEM_42
# define GT_PP_SEQ_ELEM_44(_) GT_PP_SEQ_ELEM_43
# define GT_PP_SEQ_ELEM_45(_) GT_PP_SEQ_ELEM_44
# define GT_PP_SEQ_ELEM_46(_) GT_PP_SEQ_ELEM_45
# define GT_PP_SEQ_ELEM_47(_) GT_PP_SEQ_ELEM_46
# define GT_PP_SEQ_ELEM_48(_) GT_PP_SEQ_ELEM_47
# define GT_PP_SEQ_ELEM_49(_) GT_PP_SEQ_ELEM_48
# define GT_PP_SEQ_ELEM_50(_) GT_PP_SEQ_ELEM_49
# define GT_PP_SEQ_ELEM_51(_) GT_PP_SEQ_ELEM_50
# define GT_PP_SEQ_ELEM_52(_) GT_PP_SEQ_ELEM_51
# define GT_PP_SEQ_ELEM_53(_) GT_PP_SEQ_ELEM_52
# define GT_PP_SEQ_ELEM_54(_) GT_PP_SEQ_ELEM_53
# define GT_PP_SEQ_ELEM_55(_) GT_PP_SEQ_ELEM_54
# define GT_PP_SEQ_ELEM_56(_) GT_PP_SEQ_ELEM_55
# define GT_PP_SEQ_ELEM_57(_) GT_PP_SEQ_ELEM_56
# define GT_PP_SEQ_ELEM_58(_) GT_PP_SEQ_ELEM_57
# define GT_PP_SEQ_ELEM_59(_) GT_PP_SEQ_ELEM_58
# define GT_PP_SEQ_ELEM_60(_) GT_PP_SEQ_ELEM_59
# define GT_PP_SEQ_ELEM_61(_) GT_PP_SEQ_ELEM_60
# define GT_PP_SEQ_ELEM_62(_) GT_PP_SEQ_ELEM_61
# define GT_PP_SEQ_ELEM_63(_) GT_PP_SEQ_ELEM_62
# define GT_PP_SEQ_ELEM_64(_) GT_PP_SEQ_ELEM_63
# define GT_PP_SEQ_ELEM_65(_) GT_PP_SEQ_ELEM_64
# define GT_PP_SEQ_ELEM_66(_) GT_PP_SEQ_ELEM_65
# define GT_PP_SEQ_ELEM_67(_) GT_PP_SEQ_ELEM_66
# define GT_PP_SEQ_ELEM_68(_) GT_PP_SEQ_ELEM_67
# define GT_PP_SEQ_ELEM_69(_) GT_PP_SEQ_ELEM_68
# define GT_PP_SEQ_ELEM_70(_) GT_PP_SEQ_ELEM_69
# define GT_PP_SEQ_ELEM_71(_) GT_PP_SEQ_ELEM_70
# define GT_PP_SEQ_ELEM_72(_) GT_PP_SEQ_ELEM_71
# define GT_PP_SEQ_ELEM_73(_) GT_PP_SEQ_ELEM_72
# define GT_PP_SEQ_ELEM_74(_) GT_PP_SEQ_ELEM_73
# define GT_PP_SEQ_ELEM_75(_) GT_PP_SEQ_ELEM_74
# define GT_PP_SEQ_ELEM_76(_) GT_PP_SEQ_ELEM_75
# define GT_PP_SEQ_ELEM_77(_) GT_PP_SEQ_ELEM_76
# define GT_PP_SEQ_ELEM_78(_) GT_PP_SEQ_ELEM_77
# define GT_PP_SEQ_ELEM_79(_) GT_PP_SEQ_ELEM_78
# define GT_PP_SEQ_ELEM_80(_) GT_PP_SEQ_ELEM_79
# define GT_PP_SEQ_ELEM_81(_) GT_PP_SEQ_ELEM_80
# define GT_PP_SEQ_ELEM_82(_) GT_PP_SEQ_ELEM_81
# define GT_PP_SEQ_ELEM_83(_) GT_PP_SEQ_ELEM_82
# define GT_PP_SEQ_ELEM_84(_) GT_PP_SEQ_ELEM_83
# define GT_PP_SEQ_ELEM_85(_) GT_PP_SEQ_ELEM_84
# define GT_PP_SEQ_ELEM_86(_) GT_PP_SEQ_ELEM_85
# define GT_PP_SEQ_ELEM_87(_) GT_PP_SEQ_ELEM_86
# define GT_PP_SEQ_ELEM_88(_) GT_PP_SEQ_ELEM_87
# define GT_PP_SEQ_ELEM_89(_) GT_PP_SEQ_ELEM_88
# define GT_PP_SEQ_ELEM_90(_) GT_PP_SEQ_ELEM_89
# define GT_PP_SEQ_ELEM_91(_) GT_PP_SEQ_ELEM_90
# define GT_PP_SEQ_ELEM_92(_) GT_PP_SEQ_ELEM_91
# define GT_PP_SEQ_ELEM_93(_) GT_PP_SEQ_ELEM_92
# define GT_PP_SEQ_ELEM_94(_) GT_PP_SEQ_ELEM_93
# define GT_PP_SEQ_ELEM_95(_) GT_PP_SEQ_ELEM_94
# define GT_PP_SEQ_ELEM_96(_) GT_PP_SEQ_ELEM_95
# define GT_PP_SEQ_ELEM_97(_) GT_PP_SEQ_ELEM_96
# define GT_PP_SEQ_ELEM_98(_) GT_PP_SEQ_ELEM_97
# define GT_PP_SEQ_ELEM_99(_) GT_PP_SEQ_ELEM_98
# define GT_PP_SEQ_ELEM_100(_) GT_PP_SEQ_ELEM_99
# define GT_PP_SEQ_ELEM_101(_) GT_PP_SEQ_ELEM_100
# define GT_PP_SEQ_ELEM_102(_) GT_PP_SEQ_ELEM_101
# define GT_PP_SEQ_ELEM_103(_) GT_PP_SEQ_ELEM_102
# define GT_PP_SEQ_ELEM_104(_) GT_PP_SEQ_ELEM_103
# define GT_PP_SEQ_ELEM_105(_) GT_PP_SEQ_ELEM_104
# define GT_PP_SEQ_ELEM_106(_) GT_PP_SEQ_ELEM_105
# define GT_PP_SEQ_ELEM_107(_) GT_PP_SEQ_ELEM_106
# define GT_PP_SEQ_ELEM_108(_) GT_PP_SEQ_ELEM_107
# define GT_PP_SEQ_ELEM_109(_) GT_PP_SEQ_ELEM_108
# define GT_PP_SEQ_ELEM_110(_) GT_PP_SEQ_ELEM_109
# define GT_PP_SEQ_ELEM_111(_) GT_PP_SEQ_ELEM_110
# define GT_PP_SEQ_ELEM_112(_) GT_PP_SEQ_ELEM_111
# define GT_PP_SEQ_ELEM_113(_) GT_PP_SEQ_ELEM_112
# define GT_PP_SEQ_ELEM_114(_) GT_PP_SEQ_ELEM_113
# define GT_PP_SEQ_ELEM_115(_) GT_PP_SEQ_ELEM_114
# define GT_PP_SEQ_ELEM_116(_) GT_PP_SEQ_ELEM_115
# define GT_PP_SEQ_ELEM_117(_) GT_PP_SEQ_ELEM_116
# define GT_PP_SEQ_ELEM_118(_) GT_PP_SEQ_ELEM_117
# define GT_PP_SEQ_ELEM_119(_) GT_PP_SEQ_ELEM_118
# define GT_PP_SEQ_ELEM_120(_) GT_PP_SEQ_ELEM_119
# define GT_PP_SEQ_ELEM_121(_) GT_PP_SEQ_ELEM_120
# define GT_PP_SEQ_ELEM_122(_) GT_PP_SEQ_ELEM_121
# define GT_PP_SEQ_ELEM_123(_) GT_PP_SEQ_ELEM_122
# define GT_PP_SEQ_ELEM_124(_) GT_PP_SEQ_ELEM_123
# define GT_PP_SEQ_ELEM_125(_) GT_PP_SEQ_ELEM_124
# define GT_PP_SEQ_ELEM_126(_) GT_PP_SEQ_ELEM_125
# define GT_PP_SEQ_ELEM_127(_) GT_PP_SEQ_ELEM_126
# define GT_PP_SEQ_ELEM_128(_) GT_PP_SEQ_ELEM_127
# define GT_PP_SEQ_ELEM_129(_) GT_PP_SEQ_ELEM_128
# define GT_PP_SEQ_ELEM_130(_) GT_PP_SEQ_ELEM_129
# define GT_PP_SEQ_ELEM_131(_) GT_PP_SEQ_ELEM_130
# define GT_PP_SEQ_ELEM_132(_) GT_PP_SEQ_ELEM_131
# define GT_PP_SEQ_ELEM_133(_) GT_PP_SEQ_ELEM_132
# define GT_PP_SEQ_ELEM_134(_) GT_PP_SEQ_ELEM_133
# define GT_PP_SEQ_ELEM_135(_) GT_PP_SEQ_ELEM_134
# define GT_PP_SEQ_ELEM_136(_) GT_PP_SEQ_ELEM_135
# define GT_PP_SEQ_ELEM_137(_) GT_PP_SEQ_ELEM_136
# define GT_PP_SEQ_ELEM_138(_) GT_PP_SEQ_ELEM_137
# define GT_PP_SEQ_ELEM_139(_) GT_PP_SEQ_ELEM_138
# define GT_PP_SEQ_ELEM_140(_) GT_PP_SEQ_ELEM_139
# define GT_PP_SEQ_ELEM_141(_) GT_PP_SEQ_ELEM_140
# define GT_PP_SEQ_ELEM_142(_) GT_PP_SEQ_ELEM_141
# define GT_PP_SEQ_ELEM_143(_) GT_PP_SEQ_ELEM_142
# define GT_PP_SEQ_ELEM_144(_) GT_PP_SEQ_ELEM_143
# define GT_PP_SEQ_ELEM_145(_) GT_PP_SEQ_ELEM_144
# define GT_PP_SEQ_ELEM_146(_) GT_PP_SEQ_ELEM_145
# define GT_PP_SEQ_ELEM_147(_) GT_PP_SEQ_ELEM_146
# define GT_PP_SEQ_ELEM_148(_) GT_PP_SEQ_ELEM_147
# define GT_PP_SEQ_ELEM_149(_) GT_PP_SEQ_ELEM_148
# define GT_PP_SEQ_ELEM_150(_) GT_PP_SEQ_ELEM_149
# define GT_PP_SEQ_ELEM_151(_) GT_PP_SEQ_ELEM_150
# define GT_PP_SEQ_ELEM_152(_) GT_PP_SEQ_ELEM_151
# define GT_PP_SEQ_ELEM_153(_) GT_PP_SEQ_ELEM_152
# define GT_PP_SEQ_ELEM_154(_) GT_PP_SEQ_ELEM_153
# define GT_PP_SEQ_ELEM_155(_) GT_PP_SEQ_ELEM_154
# define GT_PP_SEQ_ELEM_156(_) GT_PP_SEQ_ELEM_155
# define GT_PP_SEQ_ELEM_157(_) GT_PP_SEQ_ELEM_156
# define GT_PP_SEQ_ELEM_158(_) GT_PP_SEQ_ELEM_157
# define GT_PP_SEQ_ELEM_159(_) GT_PP_SEQ_ELEM_158
# define GT_PP_SEQ_ELEM_160(_) GT_PP_SEQ_ELEM_159
# define GT_PP_SEQ_ELEM_161(_) GT_PP_SEQ_ELEM_160
# define GT_PP_SEQ_ELEM_162(_) GT_PP_SEQ_ELEM_161
# define GT_PP_SEQ_ELEM_163(_) GT_PP_SEQ_ELEM_162
# define GT_PP_SEQ_ELEM_164(_) GT_PP_SEQ_ELEM_163
# define GT_PP_SEQ_ELEM_165(_) GT_PP_SEQ_ELEM_164
# define GT_PP_SEQ_ELEM_166(_) GT_PP_SEQ_ELEM_165
# define GT_PP_SEQ_ELEM_167(_) GT_PP_SEQ_ELEM_166
# define GT_PP_SEQ_ELEM_168(_) GT_PP_SEQ_ELEM_167
# define GT_PP_SEQ_ELEM_169(_) GT_PP_SEQ_ELEM_168
# define GT_PP_SEQ_ELEM_170(_) GT_PP_SEQ_ELEM_169
# define GT_PP_SEQ_ELEM_171(_) GT_PP_SEQ_ELEM_170
# define GT_PP_SEQ_ELEM_172(_) GT_PP_SEQ_ELEM_171
# define GT_PP_SEQ_ELEM_173(_) GT_PP_SEQ_ELEM_172
# define GT_PP_SEQ_ELEM_174(_) GT_PP_SEQ_ELEM_173
# define GT_PP_SEQ_ELEM_175(_) GT_PP_SEQ_ELEM_174
# define GT_PP_SEQ_ELEM_176(_) GT_PP_SEQ_ELEM_175
# define GT_PP_SEQ_ELEM_177(_) GT_PP_SEQ_ELEM_176
# define GT_PP_SEQ_ELEM_178(_) GT_PP_SEQ_ELEM_177
# define GT_PP_SEQ_ELEM_179(_) GT_PP_SEQ_ELEM_178
# define GT_PP_SEQ_ELEM_180(_) GT_PP_SEQ_ELEM_179
# define GT_PP_SEQ_ELEM_181(_) GT_PP_SEQ_ELEM_180
# define GT_PP_SEQ_ELEM_182(_) GT_PP_SEQ_ELEM_181
# define GT_PP_SEQ_ELEM_183(_) GT_PP_SEQ_ELEM_182
# define GT_PP_SEQ_ELEM_184(_) GT_PP_SEQ_ELEM_183
# define GT_PP_SEQ_ELEM_185(_) GT_PP_SEQ_ELEM_184
# define GT_PP_SEQ_ELEM_186(_) GT_PP_SEQ_ELEM_185
# define GT_PP_SEQ_ELEM_187(_) GT_PP_SEQ_ELEM_186
# define GT_PP_SEQ_ELEM_188(_) GT_PP_SEQ_ELEM_187
# define GT_PP_SEQ_ELEM_189(_) GT_PP_SEQ_ELEM_188
# define GT_PP_SEQ_ELEM_190(_) GT_PP_SEQ_ELEM_189
# define GT_PP_SEQ_ELEM_191(_) GT_PP_SEQ_ELEM_190
# define GT_PP_SEQ_ELEM_192(_) GT_PP_SEQ_ELEM_191
# define GT_PP_SEQ_ELEM_193(_) GT_PP_SEQ_ELEM_192
# define GT_PP_SEQ_ELEM_194(_) GT_PP_SEQ_ELEM_193
# define GT_PP_SEQ_ELEM_195(_) GT_PP_SEQ_ELEM_194
# define GT_PP_SEQ_ELEM_196(_) GT_PP_SEQ_ELEM_195
# define GT_PP_SEQ_ELEM_197(_) GT_PP_SEQ_ELEM_196
# define GT_PP_SEQ_ELEM_198(_) GT_PP_SEQ_ELEM_197
# define GT_PP_SEQ_ELEM_199(_) GT_PP_SEQ_ELEM_198
# define GT_PP_SEQ_ELEM_200(_) GT_PP_SEQ_ELEM_199
# define GT_PP_SEQ_ELEM_201(_) GT_PP_SEQ_ELEM_200
# define GT_PP_SEQ_ELEM_202(_) GT_PP_SEQ_ELEM_201
# define GT_PP_SEQ_ELEM_203(_) GT_PP_SEQ_ELEM_202
# define GT_PP_SEQ_ELEM_204(_) GT_PP_SEQ_ELEM_203
# define GT_PP_SEQ_ELEM_205(_) GT_PP_SEQ_ELEM_204
# define GT_PP_SEQ_ELEM_206(_) GT_PP_SEQ_ELEM_205
# define GT_PP_SEQ_ELEM_207(_) GT_PP_SEQ_ELEM_206
# define GT_PP_SEQ_ELEM_208(_) GT_PP_SEQ_ELEM_207
# define GT_PP_SEQ_ELEM_209(_) GT_PP_SEQ_ELEM_208
# define GT_PP_SEQ_ELEM_210(_) GT_PP_SEQ_ELEM_209
# define GT_PP_SEQ_ELEM_211(_) GT_PP_SEQ_ELEM_210
# define GT_PP_SEQ_ELEM_212(_) GT_PP_SEQ_ELEM_211
# define GT_PP_SEQ_ELEM_213(_) GT_PP_SEQ_ELEM_212
# define GT_PP_SEQ_ELEM_214(_) GT_PP_SEQ_ELEM_213
# define GT_PP_SEQ_ELEM_215(_) GT_PP_SEQ_ELEM_214
# define GT_PP_SEQ_ELEM_216(_) GT_PP_SEQ_ELEM_215
# define GT_PP_SEQ_ELEM_217(_) GT_PP_SEQ_ELEM_216
# define GT_PP_SEQ_ELEM_218(_) GT_PP_SEQ_ELEM_217
# define GT_PP_SEQ_ELEM_219(_) GT_PP_SEQ_ELEM_218
# define GT_PP_SEQ_ELEM_220(_) GT_PP_SEQ_ELEM_219
# define GT_PP_SEQ_ELEM_221(_) GT_PP_SEQ_ELEM_220
# define GT_PP_SEQ_ELEM_222(_) GT_PP_SEQ_ELEM_221
# define GT_PP_SEQ_ELEM_223(_) GT_PP_SEQ_ELEM_222
# define GT_PP_SEQ_ELEM_224(_) GT_PP_SEQ_ELEM_223
# define GT_PP_SEQ_ELEM_225(_) GT_PP_SEQ_ELEM_224
# define GT_PP_SEQ_ELEM_226(_) GT_PP_SEQ_ELEM_225
# define GT_PP_SEQ_ELEM_227(_) GT_PP_SEQ_ELEM_226
# define GT_PP_SEQ_ELEM_228(_) GT_PP_SEQ_ELEM_227
# define GT_PP_SEQ_ELEM_229(_) GT_PP_SEQ_ELEM_228
# define GT_PP_SEQ_ELEM_230(_) GT_PP_SEQ_ELEM_229
# define GT_PP_SEQ_ELEM_231(_) GT_PP_SEQ_ELEM_230
# define GT_PP_SEQ_ELEM_232(_) GT_PP_SEQ_ELEM_231
# define GT_PP_SEQ_ELEM_233(_) GT_PP_SEQ_ELEM_232
# define GT_PP_SEQ_ELEM_234(_) GT_PP_SEQ_ELEM_233
# define GT_PP_SEQ_ELEM_235(_) GT_PP_SEQ_ELEM_234
# define GT_PP_SEQ_ELEM_236(_) GT_PP_SEQ_ELEM_235
# define GT_PP_SEQ_ELEM_237(_) GT_PP_SEQ_ELEM_236
# define GT_PP_SEQ_ELEM_238(_) GT_PP_SEQ_ELEM_237
# define GT_PP_SEQ_ELEM_239(_) GT_PP_SEQ_ELEM_238
# define GT_PP_SEQ_ELEM_240(_) GT_PP_SEQ_ELEM_239
# define GT_PP_SEQ_ELEM_241(_) GT_PP_SEQ_ELEM_240
# define GT_PP_SEQ_ELEM_242(_) GT_PP_SEQ_ELEM_241
# define GT_PP_SEQ_ELEM_243(_) GT_PP_SEQ_ELEM_242
# define GT_PP_SEQ_ELEM_244(_) GT_PP_SEQ_ELEM_243
# define GT_PP_SEQ_ELEM_245(_) GT_PP_SEQ_ELEM_244
# define GT_PP_SEQ_ELEM_246(_) GT_PP_SEQ_ELEM_245
# define GT_PP_SEQ_ELEM_247(_) GT_PP_SEQ_ELEM_246
# define GT_PP_SEQ_ELEM_248(_) GT_PP_SEQ_ELEM_247
# define GT_PP_SEQ_ELEM_249(_) GT_PP_SEQ_ELEM_248
# define GT_PP_SEQ_ELEM_250(_) GT_PP_SEQ_ELEM_249
# define GT_PP_SEQ_ELEM_251(_) GT_PP_SEQ_ELEM_250
# define GT_PP_SEQ_ELEM_252(_) GT_PP_SEQ_ELEM_251
# define GT_PP_SEQ_ELEM_253(_) GT_PP_SEQ_ELEM_252
# define GT_PP_SEQ_ELEM_254(_) GT_PP_SEQ_ELEM_253
# define GT_PP_SEQ_ELEM_255(_) GT_PP_SEQ_ELEM_254
#
# else
#
# include <gridtools/preprocessor/config/limits.hpp>
#
# if GT_PP_LIMIT_SEQ == 256
# include <gridtools/preprocessor/seq/limits/elem_256.hpp>
# elif GT_PP_LIMIT_SEQ == 512
# include <gridtools/preprocessor/seq/limits/elem_256.hpp>
# include <gridtools/preprocessor/seq/limits/elem_512.hpp>
# elif GT_PP_LIMIT_SEQ == 1024
# include <gridtools/preprocessor/seq/limits/elem_256.hpp>
# include <gridtools/preprocessor/seq/limits/elem_512.hpp>
# include <gridtools/preprocessor/seq/limits/elem_1024.hpp>
# else
# error Incorrect value for the GT_PP_LIMIT_SEQ limit
# endif
#
# endif
#
# endif
