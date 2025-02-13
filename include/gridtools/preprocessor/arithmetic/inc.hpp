# /* Copyright (C) 2001
#  * Housemarque Oy
#  * http://www.housemarque.com
#  *
#  * Distributed under the Boost Software License, Version 1.0. (See
#  * accompanying file LICENSE_1_0.txt or copy at
#  * http://www.boost.org/LICENSE_1_0.txt)
#  */
#
# /* Revised by Paul Mensonides (2002) */
# /* Revised by Edward Diener (2020) */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_ARITHMETIC_INC_HPP
# define GT_PREPROCESSOR_ARITHMETIC_INC_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_STRICT()
#
# /* GT_PP_INC */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC()
#    define GT_PP_INC(x) GT_PP_INC_I(x)
# else
#    define GT_PP_INC(x) GT_PP_INC_OO((x))
#    define GT_PP_INC_OO(par) GT_PP_INC_I ## par
# endif
#
# define GT_PP_INC_I(x) GT_PP_INC_ ## x
#
# define GT_PP_INC_0 1
# define GT_PP_INC_1 2
# define GT_PP_INC_2 3
# define GT_PP_INC_3 4
# define GT_PP_INC_4 5
# define GT_PP_INC_5 6
# define GT_PP_INC_6 7
# define GT_PP_INC_7 8
# define GT_PP_INC_8 9
# define GT_PP_INC_9 10
# define GT_PP_INC_10 11
# define GT_PP_INC_11 12
# define GT_PP_INC_12 13
# define GT_PP_INC_13 14
# define GT_PP_INC_14 15
# define GT_PP_INC_15 16
# define GT_PP_INC_16 17
# define GT_PP_INC_17 18
# define GT_PP_INC_18 19
# define GT_PP_INC_19 20
# define GT_PP_INC_20 21
# define GT_PP_INC_21 22
# define GT_PP_INC_22 23
# define GT_PP_INC_23 24
# define GT_PP_INC_24 25
# define GT_PP_INC_25 26
# define GT_PP_INC_26 27
# define GT_PP_INC_27 28
# define GT_PP_INC_28 29
# define GT_PP_INC_29 30
# define GT_PP_INC_30 31
# define GT_PP_INC_31 32
# define GT_PP_INC_32 33
# define GT_PP_INC_33 34
# define GT_PP_INC_34 35
# define GT_PP_INC_35 36
# define GT_PP_INC_36 37
# define GT_PP_INC_37 38
# define GT_PP_INC_38 39
# define GT_PP_INC_39 40
# define GT_PP_INC_40 41
# define GT_PP_INC_41 42
# define GT_PP_INC_42 43
# define GT_PP_INC_43 44
# define GT_PP_INC_44 45
# define GT_PP_INC_45 46
# define GT_PP_INC_46 47
# define GT_PP_INC_47 48
# define GT_PP_INC_48 49
# define GT_PP_INC_49 50
# define GT_PP_INC_50 51
# define GT_PP_INC_51 52
# define GT_PP_INC_52 53
# define GT_PP_INC_53 54
# define GT_PP_INC_54 55
# define GT_PP_INC_55 56
# define GT_PP_INC_56 57
# define GT_PP_INC_57 58
# define GT_PP_INC_58 59
# define GT_PP_INC_59 60
# define GT_PP_INC_60 61
# define GT_PP_INC_61 62
# define GT_PP_INC_62 63
# define GT_PP_INC_63 64
# define GT_PP_INC_64 65
# define GT_PP_INC_65 66
# define GT_PP_INC_66 67
# define GT_PP_INC_67 68
# define GT_PP_INC_68 69
# define GT_PP_INC_69 70
# define GT_PP_INC_70 71
# define GT_PP_INC_71 72
# define GT_PP_INC_72 73
# define GT_PP_INC_73 74
# define GT_PP_INC_74 75
# define GT_PP_INC_75 76
# define GT_PP_INC_76 77
# define GT_PP_INC_77 78
# define GT_PP_INC_78 79
# define GT_PP_INC_79 80
# define GT_PP_INC_80 81
# define GT_PP_INC_81 82
# define GT_PP_INC_82 83
# define GT_PP_INC_83 84
# define GT_PP_INC_84 85
# define GT_PP_INC_85 86
# define GT_PP_INC_86 87
# define GT_PP_INC_87 88
# define GT_PP_INC_88 89
# define GT_PP_INC_89 90
# define GT_PP_INC_90 91
# define GT_PP_INC_91 92
# define GT_PP_INC_92 93
# define GT_PP_INC_93 94
# define GT_PP_INC_94 95
# define GT_PP_INC_95 96
# define GT_PP_INC_96 97
# define GT_PP_INC_97 98
# define GT_PP_INC_98 99
# define GT_PP_INC_99 100
# define GT_PP_INC_100 101
# define GT_PP_INC_101 102
# define GT_PP_INC_102 103
# define GT_PP_INC_103 104
# define GT_PP_INC_104 105
# define GT_PP_INC_105 106
# define GT_PP_INC_106 107
# define GT_PP_INC_107 108
# define GT_PP_INC_108 109
# define GT_PP_INC_109 110
# define GT_PP_INC_110 111
# define GT_PP_INC_111 112
# define GT_PP_INC_112 113
# define GT_PP_INC_113 114
# define GT_PP_INC_114 115
# define GT_PP_INC_115 116
# define GT_PP_INC_116 117
# define GT_PP_INC_117 118
# define GT_PP_INC_118 119
# define GT_PP_INC_119 120
# define GT_PP_INC_120 121
# define GT_PP_INC_121 122
# define GT_PP_INC_122 123
# define GT_PP_INC_123 124
# define GT_PP_INC_124 125
# define GT_PP_INC_125 126
# define GT_PP_INC_126 127
# define GT_PP_INC_127 128
# define GT_PP_INC_128 129
# define GT_PP_INC_129 130
# define GT_PP_INC_130 131
# define GT_PP_INC_131 132
# define GT_PP_INC_132 133
# define GT_PP_INC_133 134
# define GT_PP_INC_134 135
# define GT_PP_INC_135 136
# define GT_PP_INC_136 137
# define GT_PP_INC_137 138
# define GT_PP_INC_138 139
# define GT_PP_INC_139 140
# define GT_PP_INC_140 141
# define GT_PP_INC_141 142
# define GT_PP_INC_142 143
# define GT_PP_INC_143 144
# define GT_PP_INC_144 145
# define GT_PP_INC_145 146
# define GT_PP_INC_146 147
# define GT_PP_INC_147 148
# define GT_PP_INC_148 149
# define GT_PP_INC_149 150
# define GT_PP_INC_150 151
# define GT_PP_INC_151 152
# define GT_PP_INC_152 153
# define GT_PP_INC_153 154
# define GT_PP_INC_154 155
# define GT_PP_INC_155 156
# define GT_PP_INC_156 157
# define GT_PP_INC_157 158
# define GT_PP_INC_158 159
# define GT_PP_INC_159 160
# define GT_PP_INC_160 161
# define GT_PP_INC_161 162
# define GT_PP_INC_162 163
# define GT_PP_INC_163 164
# define GT_PP_INC_164 165
# define GT_PP_INC_165 166
# define GT_PP_INC_166 167
# define GT_PP_INC_167 168
# define GT_PP_INC_168 169
# define GT_PP_INC_169 170
# define GT_PP_INC_170 171
# define GT_PP_INC_171 172
# define GT_PP_INC_172 173
# define GT_PP_INC_173 174
# define GT_PP_INC_174 175
# define GT_PP_INC_175 176
# define GT_PP_INC_176 177
# define GT_PP_INC_177 178
# define GT_PP_INC_178 179
# define GT_PP_INC_179 180
# define GT_PP_INC_180 181
# define GT_PP_INC_181 182
# define GT_PP_INC_182 183
# define GT_PP_INC_183 184
# define GT_PP_INC_184 185
# define GT_PP_INC_185 186
# define GT_PP_INC_186 187
# define GT_PP_INC_187 188
# define GT_PP_INC_188 189
# define GT_PP_INC_189 190
# define GT_PP_INC_190 191
# define GT_PP_INC_191 192
# define GT_PP_INC_192 193
# define GT_PP_INC_193 194
# define GT_PP_INC_194 195
# define GT_PP_INC_195 196
# define GT_PP_INC_196 197
# define GT_PP_INC_197 198
# define GT_PP_INC_198 199
# define GT_PP_INC_199 200
# define GT_PP_INC_200 201
# define GT_PP_INC_201 202
# define GT_PP_INC_202 203
# define GT_PP_INC_203 204
# define GT_PP_INC_204 205
# define GT_PP_INC_205 206
# define GT_PP_INC_206 207
# define GT_PP_INC_207 208
# define GT_PP_INC_208 209
# define GT_PP_INC_209 210
# define GT_PP_INC_210 211
# define GT_PP_INC_211 212
# define GT_PP_INC_212 213
# define GT_PP_INC_213 214
# define GT_PP_INC_214 215
# define GT_PP_INC_215 216
# define GT_PP_INC_216 217
# define GT_PP_INC_217 218
# define GT_PP_INC_218 219
# define GT_PP_INC_219 220
# define GT_PP_INC_220 221
# define GT_PP_INC_221 222
# define GT_PP_INC_222 223
# define GT_PP_INC_223 224
# define GT_PP_INC_224 225
# define GT_PP_INC_225 226
# define GT_PP_INC_226 227
# define GT_PP_INC_227 228
# define GT_PP_INC_228 229
# define GT_PP_INC_229 230
# define GT_PP_INC_230 231
# define GT_PP_INC_231 232
# define GT_PP_INC_232 233
# define GT_PP_INC_233 234
# define GT_PP_INC_234 235
# define GT_PP_INC_235 236
# define GT_PP_INC_236 237
# define GT_PP_INC_237 238
# define GT_PP_INC_238 239
# define GT_PP_INC_239 240
# define GT_PP_INC_240 241
# define GT_PP_INC_241 242
# define GT_PP_INC_242 243
# define GT_PP_INC_243 244
# define GT_PP_INC_244 245
# define GT_PP_INC_245 246
# define GT_PP_INC_246 247
# define GT_PP_INC_247 248
# define GT_PP_INC_248 249
# define GT_PP_INC_249 250
# define GT_PP_INC_250 251
# define GT_PP_INC_251 252
# define GT_PP_INC_252 253
# define GT_PP_INC_253 254
# define GT_PP_INC_254 255
# define GT_PP_INC_255 256
# define GT_PP_INC_256 256
#
# else
#
# /* GT_PP_INC */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC()
#    define GT_PP_INC(x) GT_PP_INC_I(x)
# else
#    define GT_PP_INC(x) GT_PP_INC_OO((x))
#    define GT_PP_INC_OO(par) GT_PP_INC_I ## par
# endif
#
# define GT_PP_INC_I(x) GT_PP_INC_ ## x
#
# include <gridtools/preprocessor/config/limits.hpp>
#
# if GT_PP_LIMIT_MAG == 256
# include <gridtools/preprocessor/arithmetic/limits/inc_256.hpp>
# elif GT_PP_LIMIT_MAG == 512
# include <gridtools/preprocessor/arithmetic/limits/inc_256.hpp>
# include <gridtools/preprocessor/arithmetic/limits/inc_512.hpp>
# elif GT_PP_LIMIT_MAG == 1024
# include <gridtools/preprocessor/arithmetic/limits/inc_256.hpp>
# include <gridtools/preprocessor/arithmetic/limits/inc_512.hpp>
# include <gridtools/preprocessor/arithmetic/limits/inc_1024.hpp>
# else
# error Incorrect value for the GT_PP_LIMIT_MAG limit
# endif
#
# endif
#
# endif
