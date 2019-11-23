
import numpy as np
import pandas as pd

labelsmuc = pd.read_csv('./mudi_labels.csv', header=None).values[:,2]#6614*2=13228
labelsmui = pd.read_csv('./mudi_labels.csv', header=None).values[:,4]#6614*2=13228
labelsmec = pd.read_csv('./mescalina_labels.csv', header=None).values[:,4]#3216+6077=9293+2861=12154
labelsmei = pd.read_csv('./mescalina_labels.csv', header=None).values[:,2]#3216+6077=9293+2861=12154
labelsurb = pd.read_csv('./urban_labels.csv', header=None).values[:,2]#8732*2=17464


mudi = pd.read_csv('./mudi_AugmentedNoise.csv', header=None).as_matrix() #6555
mudi = mudi[:,1:6553] #6552


labelsmuc = np.concatenate([labelsmuc,labelsmuc])#.astype(str)#6553
#ranked=[0, 4, 11, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 40, 41, 43, 46, 47, 51, 56, 57, 59, 60, 61, 64, 65, 66, 70, 72, 73, 74, 75, 76, 77, 79, 82, 85, 86, 90, 95, 96, 98, 99, 100, 103, 104, 105, 109, 111, 112, 113, 114, 115, 116, 118, 121, 124, 125, 126, 129, 133, 134, 137, 138, 139, 142, 150, 151, 152, 153, 154, 155, 159, 163, 168, 176, 177, 178, 182, 183, 184, 187, 189, 190, 191, 192, 193, 194, 202, 207, 215, 216, 217, 221, 222, 223, 226, 228, 229, 231, 233, 241, 254, 255, 256, 260, 261, 265, 267, 268, 270, 276, 280, 285, 293, 294, 295, 299, 300, 304, 306, 307, 309, 332, 333, 334, 338, 339, 340, 343, 345, 346, 347, 348, 349, 350, 371, 372, 373, 377, 378, 379, 382, 384, 385, 386, 387, 388, 389, 410, 411, 412, 416, 417, 421, 423, 424, 426, 449, 450, 451, 455, 456, 460, 462, 463, 465, 488, 489, 490, 501, 508, 511, 515, 516, 518, 524, 527, 528, 529, 530, 531, 532, 533, 534, 540, 541, 542, 543, 544, 545, 678, 717, 731, 732, 756, 795, 834, 859, 885, 898, 901, 914, 923, 924, 928, 937, 944, 953, 962, 963, 967, 1263, 1341, 1380, 1497, 1536, 1565, 1580, 1581, 1598, 1600, 1603, 1604, 1606, 1607, 1608, 1611, 1614, 1615, 1616, 1619, 1620, 1621, 1624, 1625, 1626, 1630, 1632, 1633, 1634, 1635, 1636, 1637, 1639, 1643, 1653, 1654, 1655, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1669, 1671, 1672, 1673, 1677, 1678, 1681, 1682, 1684, 1688, 1693, 1694, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1708, 1710, 1711, 1712, 1713, 1714, 1715, 1770, 1809, 1848, 1875, 1914, 1951, 1954, 1965, 1972, 1976, 1977, 1981, 1988, 1990, 1992, 1993, 1994, 1997, 1998, 2001, 2004, 2005, 2006, 2007, 2009, 2010, 2011, 2014, 2015, 2016, 2020, 2022, 2023, 2025, 2027, 2043, 2070, 2071, 2085, 2086, 2087, 2088, 2089, 2105, 2189, 2192, 2201, 2202, 2204, 2205, 2207, 2209, 2218, 2220, 2222, 2240, 2263, 2267, 2280, 2306, 2345, 2692, 2695, 2696, 2699, 2707, 2708, 2709, 2711, 2712, 2714, 2715, 2716, 2723, 2725, 2726, 2727, 2728, 2729, 2904, 2943, 2982, 2993, 3006, 3031, 3043, 3069, 3070, 3082, 3085, 3094, 3098, 3099, 3107, 3108, 3109, 3111, 3137, 3149, 3150, 3216, 3227, 3255, 3294, 3333, 3579, 3800, 3809, 3810, 3846, 3850, 3868, 3873, 3876, 3889, 3900, 3968, 3969, 4007, 4008, 4138, 4139, 4160, 4161, 4174, 4175, 4177, 4178, 4185, 4190, 4191, 4193, 4199, 4200, 4202, 4203, 4211, 4230, 4241, 4256, 4269, 4270, 4281, 4289, 4376, 4388, 4390, 4391, 4392, 4393, 4402, 4404, 4406, 4412, 4424, 4447, 4450, 4463, 4464, 4472, 4473, 4880, 4883, 4892, 4897, 4899, 4900, 4903, 4909, 4910, 4911, 4912, 4913, 5072, 5073, 5111, 5150, 5228, 5278, 5291, 5332, 5345, 5384, 6034, 6073, 6322, 6340, 6342, 6343, 6344, 6345, 6348, 6349, 6350, 6358, 6361, 6362, 6377, 6380, 6382, 6383, 6384, 6387, 6395, 6398, 6454, 6457, 6460, 6462, 6465, 6473]
ranked=[810,771,4398,4905,732,2710,4047,693,966,2709,252,94,849,6075,299,916,309,115,261,260,495,4876,222,4904,4906,213,221,4907,4880,1161,297,295,494,98,103,101,2696,1033,915,153,2204,4879,2207,4400,994,2209,155,1625,298,1032,4399,1561,862,545,300,486,201,1626,859,369,134,3059,993,290,338,5958,202,2711,2716,2714,311,3202,95,116,4369,3969,114,3082,4373,2189,4008,901,1072,898,543,447,3852,877,1657,279,3043,3098,885,890,3021,1600,2241,310,4894,2015,413,927,410,137,889,654,465,3046,2203,3225,1664,472,5266,1071,3100,1083,330,5383,2266,207,142,5399,77,876,3076,508,1018,3199,4410,29,2982,4230,3050,5270,125,449,5282,5387,140,544,1015,5243,85,809,2721,3055,2016,59,5230,436,238,199,3115,64,280,5269,75,452,534,434,55,176,122,1951,5227,339,2005,285,3085,181,3774,192,226,56,2289,179,3094,435,113,5390,5416,3215,5395,3069,3086,2006,924,474,5386,533,62,304,3089,396,2717,928,462,955,1852,475,1607,446,929,1434,1954,3047,76,124,357,45,875,4216,1635,5292,51,5409,1990,1785,504,277,3203,480,319,2046,5231,2729,3061,3163,1977,5253,3216,1636,5150,3108,467,173,147,866,884,979,415,976,183,4206,303,408,1824,4901,154,2280,3178,5344,681,1045,3206,3255,3232,3078,5360,397,345,3211,2725,954,1656,1422,531,123,2618,1005,5347,3068,118,2001,1813,682,3160,5356,5351,5239,5377,4913,5260,441,1228,5234,162,5278,194,133,499,3128,1608,358,2047,5273,163,3176,1501,5348,168,1774,318,3154,437,46,92,84,1122,501,5299,871,1613,914,564,3186,3172,289,937,1851,1041,3133,1603,615,4239,1500,5370,965,3177,130,4798,4203,1616,356,1031,4802,1046,377,2295,3167,3193,1227,2943,4892,4841,2272,417,281,940,3217,1111,454,1997,759,3060,2614,933,426,473,1057,642,1054,1993,395,228,6397,888,2972,905,4406,963,4035,47,2267,6413,1976,1423,126,5257,206,5258,305,2977,129,3051,4229,1002,402,4231,3077,3058,485,463,3056,286,2983,1306,4167,428,4192,1615,3164,174,2998,306,529,1080,3117,2273,2630,1461,4814,4213,5189,442,532,1070,307,3137,5540,1305,5414,182,5413,87,1006,5174,4889,903,348,3529,3245,1612,86,1110,1022,161,3271,838,3107,90,1007,5262,3234,4884,1084,760,6398,4891,5252,1189,4910,6401,4837,3411,5369,1384,387,5161,992,1188,5401,3124,5156,5308,3333,1995,5182,1708,1061,820,1383,212,2013,923,40,3250,5418,4170,31,2990,2263,3254,4270,6430,3489,781,264,837,5312,2944,4225,1085,5317,3372,5184,3029,456,5362,2579]
best500=[]
for column in ranked:
    best500.append(mudi[:,column-1])
best500 = (np.asarray(best500).astype(str)).T
best500 = np.column_stack((best500,labelsmuc))

np.savetxt('mudi_context_AugmentedNoise_LLDs.csv',best500,delimiter=",",fmt="%s")
del best500

labelsmui = np.concatenate([labelsmui,labelsmui])#.astype(str)#6553
ranked=[810,771,4398,4905,849,859,862,732,3059,299,885,221,2710,338,309,3076,889,3050,890,3055,222,330,693,3043,3046,6075,875,866,901,884,898,4047,495,2709,295,3082,153,339,494,3202,1626,3069,297,213,290,134,447,5230,1625,300,3098,5243,94,1018,871,1664,201,1015,155,115,202,877,3225,298,311,1657,5266,1600,369,916,4906,261,260,4907,5227,3078,5239,279,3085,3068,5260,5270,5234,1785,3115,928,1824,5282,924,3199,319,310,5269,929,5383,3094,486,4400,5253,5399,101,434,1045,3215,345,4399,876,3211,865,3089,3206,3061,98,103,4904,125,4876,3232,5387,3108,280,915,318,1031,3086,137,864,226,3051,252,122,1656,3077,3058,207,994,5390,1041,116,5395,5386,5416,142,5292,5231,4203,3047,3056,1046,966,2696,436,4880,199,1083,285,914,888,1022,114,4879,5958,435,993,654,357,3052,140,545,304,1033,181,3053,3852,176,5262,933,3774,29,179,905,1032,5252,5409,95,462,5278,118,894,277,5273,927,124,3234,1072,4270,5299,3208,472,1020,1027,2579,1021,396,4894,4369,1561,870,413,358,3203,3100,2189,410,2203,1071,130,3117,3054,1057,2618,441,4192,1054,903,474,1050,3209,1026,192,133,543,123,154,4373,475,3163,809,356,55,923,4216,1047,1048,1019,5344,979,289,1080,976,4230,5360,1176,1935,228,3178,77,892,891,863,324,1902,85,465,397,2266,316,5347,1177,1084,5348,194,3271,4239,281,872,3160,75,3245,4837,306,2575,5356,5351,1070,307,1603,3176,5377,4763,126,455,1040,2130,3186,867,473,238,4775,1085,3172,76,377,56,480,4229,1665,1607,4840,544,1204,1203,1175,5257,5258,2015,3364,501,2591,343,332,62,3250,3128,2656,3254,499,3193,3154,3365,1852,59,3167,904,909,5418,1182,6397,937,449,408,4798,1613,1608,64,1933,3210,2289,1936,129,5291,1061,3133,3091,2614,1206,5414,113,395,2131,504,2128,5413,456,4410,131,2105,1178,212,4853,1636,5370,446,90,452,2013,186,681,940,1878,2467,682,6401,2653,392,4225,910,2016,4213,335,1044,183,286,420,2032,219,2029,4153,4220,231,6413,417,3107,963,227,292,84,1066,1002,1635,2005,4794,151,407,402,6383,5297,2241,206,305,4246,1896,2046,3092,1859,5296,1616,4281,426,1889,415,173,485,3021,163,2103,3164,4802,5264,3137,1879,955,51,3080,162,2630,2055,5362,2091,5301,45,4209,3238,1974,1669,4181,1612,1196,3490,2468,4759,1006,508,387,4814,1951,3177,6398,5369,1655,2470,1187,1954,1990,5434,5308,1221,615,1007,2280,5429,3124,992,2006,3060,4170,27,4841,2669,1615,5455,5305,5261,5242,5235,2314,5240,2295,533,47]
best500=[]
for column in ranked:
    best500.append(mudi[:,column-1])
best500 = (np.asarray(best500).astype(str)).T
best500 = np.column_stack((best500,labelsmui))

np.savetxt('mudi_individ_AugmentedNoise_LLDs.csv',best500,delimiter=",",fmt="%s")
del best500
del mudi


#####################################################################################################################

mescalina = pd.read_csv('./mescalina_AugmentedNoise.csv', header=None).as_matrix() #6555
mescalina = mescalina[:,1:6553] #6552

ranked=[155,732,153,1317,1785,2058,1863,115,1824,1356,1239,1434,1044,1200,194,927,1161,1278,1395,113,966,151,849,192,1473,465,810,1512,99,1746,888,236,1122,108,150,426,137,119,190,98,2711,4349,771,114,377,4410,4352,314,1083,537,30,77,176,158,462,138,463,2204,116,124,384,75,387,222,228,221,378,3852,928,1005,428,40,112,202,56,2241,306,4354,389,654,450,6075,910,31,177,280,1551,111,373,154,466,100,467,576,293,4449,79,95,2011,43,3708,424,410,875,44,189,4905,836,914,216,59,781,455,193,198,436,2015,1664,52,385,60,2041,423,217,392,859,855,2281,896,820,2228,73,2166,311,451,449,693,456,845,885,345,94,333,2280,91,807,454,1002,148,857,543,268,884,2723,42,2215,226,2420,6535,2968,76,823,2045,545,905,806,797,2811,372,2036,231,976,866,846,898,2016,69,5099,5891,2303,2053,152,411,2257,432,187,937,923,504,2722,460,1001,544,294,1006,6537,742,4901,2717,270,3891,2850,944,953,6538,992,827,2231,979,5149,41,871,3147,2216,191,4448,768,464,272,967,2240,3069,758,962,427,103,83,309,5879,963,4353,55,2236,924,4351,215,841,2296,901,299,784,506,2242,2270,354,178,824,852,853,350,832,315,2267,3124,348,3046,2952,988,929,2279,4907,949,295,2210,5606,275,72,1665,2991,332,4906,4239,2965,358,2019,439,502,90,173,371,2576,6551,139,233,382,353,82,3007,2298,1109,4229,862,26,788,2460,1485,889,2052,3163,2259,5153,343,1704,767,850,745,197,894,2889,300,267,229,2498,74,415,470,1678,2275,811,2718,1245,772,5684,437,109,3059,446,4058,2169,397,972,2981,1703,129,1007,46,2188,2226,319,47,1100,2695,144,1105,346,5177,1256,3020,307,754,880,1169,983,2066,4225,2733,2014,2211,5175,3030,334,255,1015,3085,4220,1846,840,2054,2772,2915,304,5089,3078,3043,1407,2929,1031,86,339,85,3039,1247,163,5723,1841,4213,1639,5165,5138,92,749,4412,1523,940,1976,64,3186,143,61,3004,5253,251,844,1227,4365,1011,1977,407,4248,4246,4216,3656,2721,53,1052,777,1018,2167,958,425,3630,527,3108,2725,101,421,1119,941,51,338,4192,110,5152,615,261,237,120,2245,2207,1261,2097,3011,3445,3440,2967,1980,1041,3016,1446,201,440,206,5230,532,4238,3537,1295,969,970,475,2972,2165,519,1050,2977,2055,368,134,891,892,1093,3160,471,5174,2064,3894,2002,3037,6549,933,3055,5188,2289,4292,207,5227,5801,1954,89,25,1265,3466,2168,1070,142,5840,20,205,2714,388]
mec1= labelsmec[:3126]
mec2= labelsmec
mec3= labelsmec[3126:]
labelsmec = np.concatenate([mec1,mec2,mec3]).astype(int)
best500=[]
for column in ranked:
    best500.append(mescalina[:,column-1])
best500 = (np.asarray(best500).astype(str)).T
best500 =  np.column_stack((best500,labelsmec))
np.savetxt('mescalina_context_AguementedNoise_LLDs.csv',best500,delimiter=",",fmt="%s")
del best500


ranked =[155,153,77,151,150,4349,4352,75,137,194,138,192,4354,115,99,1044,56,73,781,98,119,426,40,113,124,836,176,60,59,3852,222,190,221,807,76,228,820,6075,43,2968,1005,823,845,428,797,2058,114,44,4353,4351,202,55,846,806,1841,1863,72,784,1200,148,2045,1664,2241,108,4410,2011,1846,5149,177,154,217,116,6538,2036,832,462,465,158,827,1083,1582,112,455,20,1256,2965,144,216,2204,6535,61,2711,74,193,2991,928,111,139,424,69,189,451,377,927,143,268,1245,2981,850,888,384,2054,910,456,2166,855,504,272,1002,914,2041,6537,788,527,25,52,100,1580,966,1785,314,410,2087,2053,5175,436,1824,1161,95,2165,270,506,31,976,46,5153,378,1001,152,1015,3020,811,47,26,427,1031,1317,1265,2064,1278,1261,5165,306,1585,23,4905,345,1665,3007,1226,898,350,2257,1356,2231,2228,4365,280,849,1850,1222,1041,348,333,226,1598,532,187,2055,2972,1584,2009,2977,3163,923,463,373,2170,267,3030,450,5152,236,875,3440,979,4449,502,91,94,30,1869,1239,460,3445,3039,537,1704,79,824,423,992,3004,1006,64,191,1678,173,1109,859,1970,885,6551,2236,1217,255,742,1703,3466,1274,924,793,1434,1395,1242,1243,1214,5174,853,852,215,2952,2050,2998,1018,3468,6549,905,1859,1040,3011,2259,83,1210,41,3016,86,467,1050,332,2014,27,1931,768,901,4907,530,103,1639,411,1070,937,1893,1022,758,1100,1892,2052,4363,4220,387,4906,2029,70,4014,1802,1763,3975,884,346,6533,841,4225,3037,4901,2296,2298,2126,1840,2240,1011,1235,4246,4229,3458,4005,988,3966,3029,1807,2033,1768,5188,4248,1045,2061,2062,1236,2245,6536,1811,6547,1772,2270,466,6521,2065,1119,3186,2303,3995,1512,851,3956,810,6526,425,1105,745,3708,1699,372,198,1249,1295,178,3864,549,963,1066,944,129,299,53,1054,4030,866,261,953,3225,3000,3160,5204,2279,293,254,1093,2929,2105,1908,1906,1905,754,1122,163,4337,962,2010,871,231,1838,1867,1866,2242,1290,1596,1594,1593,294,967,1795,1756,1213,1284,172,319,2015,4239,3848,2044,983,1080,3991,3952,1495,3085,2275,3008,5223,389,1300,265,767,1275,2215,816,1932,862,2969,2725,1027,2103,3986,4342,3947,2420,4025,1118,2169,510,1473,1378,4053,1047,1048,1972,5184,1698,4238,300,6045,5213,1019,1304,1323,2267,3059,3894,343,2037,812,2216,3147,2723,3891,358,3069,89,980,3982,6084,5214,3943,92,145,1008,1009,2207,475,3078,4448,1079,6034,260,1061,5161,1407,3202,1007,1576,392,5156,4367,889,3176,3537,4012,3973,4043,295]
mei1= labelsmei[:3126]
mei2= labelsmei
mei3= labelsmei[3126:]
labelsmei = np.concatenate([mei1,mei2,mei3]).astype(int)
best500=[]
for column in ranked:
    best500.append(mescalina[:,column-1])
best500 = (np.asarray(best500).astype(str)).T
best500 =  np.column_stack((best500,labelsmei))
np.savetxt('mescalina_individ_AguementedNoise_LLDs.csv',best500,delimiter=",",fmt="%s")
del mescalina
del best500

############################################################################################################################


ranked =[1626,6036,1705,1716,1625,2165,1639,1665,1704,2024,2026,1703,1678,30,2009,2183,1664,3823,6084,3826,1699,2182,2180,2027,1692,1970,1713,3894,3827,1715,2023,2025,2022,2010,1988,2126,2127,6045,3900,1635,1987,1985,537,1971,2128,3852,2144,6073,6075,1642,1986,1983,1984,1710,1711,6034,3849,2166,1698,1674,2142,2140,2139,1931,2179,2178,2181,1682,2132,3889,1621,3878,1666,3888,1949,2146,2133,1677,3865,1600,1932,1676,1972,3861,1630,6011,6062,572,1944,1947,1945,1587,114,2011,112,1933,2107,508,1586,1660,1632,1633,6007,568,573,1679,2087,3,98,1620,1637,6049,3869,116,6046,3887,1937,2143,2141,1655,1653,6042,2137,190,567,6072,192,1976,194,1938,1863,3874,577,510,1582,2149,579,580,2175,2058,1951,155,584,607,4334,1636,582,2088,1990,1714,2101,2100,3850,1948,1946,1977,4365,3862,153,1836,618,619,1899,623,2153,111,178,3858,2105,1,99,1898,621,189,2163,2103,177,73,2162,2046,1892,75,606,77,2158,151,2015,6044,2164,566,605,4359,1669,1910,1634,612,2213,611,29,1893,4046,1981,1697,1912,1712,6230,3895,4363,182,6010,187,1894,1619,3839,150,100,176,6549,3848,6050,2089,2720,2093,1906,1908,1905,139,5918,2113,1942,1056,1524,2094,4337,4342,2016,939,1596,2110,4346,1581,547,1594,1593,6547,536,2102,2104,1561,542,6539,183,2020,4354,6526,6521,4330,1329,138,3800,1563,2150,4349,1996,1873,6530,1957,559,978,4915,744,1598,137,143,527,4007,2186,1918,2170,6033,861,1603,6053,6058,822,1675,2123,2214,900,1407,2047,1290,3695,2757,6533,4954,4397,544,1671,1672,4904,3656,2098,3968,3817,1785,104,3929,558,1862,6191,6023,553,3734,6517,3791,4352,3860,1708,3796,1694,105,3892,2176,1251,1962,1903,563,4351,616,1824,1134,3617,144,4353,6538,1646,2731,1758,3868,1368,1017,2734,6079,646,4291,1095,1550,148,6514,2048,2068,4241,184,2721,1485,5879,697,696,735,736,1212,3896,2770,2756,1612,2733,3784,4919,1173,1643,3383,586,699,783,4355,2114,85,738,59,2735,3853,685,548,2001,1433,1199,60,163,2049,6540,701,4295,740,6035,4931,2066,4329,658,657,2057,6152,1784,80,2111,61,4294,1797,2747,724,3835,109,1967,4940,6081,223,4957,554,2118,722,660,4943,6113,6425,2693,684,72,3830,662,2006,689,690,1616,4315,3188,4324,5801,6508,878,1651,1823,262,550,6487,2764,1121,6482,645,2759,1606,1511,6032,1238,705,723,3854,4317,4307,6510,1446,4298,215,267,124,2766,3500,3461,2119,1928,2738,1923,524,79,2074,592,2768,4303,217,1879,4958,1355,6551,1004,655,3893,2743,270,65,6475,4366,6491,6535,3891,3344,6399,76,1659,66,228,2154]

urban = pd.read_csv('./urban_AugmentedNoise.csv', header=None).as_matrix() #6555
urban = urban[:,1:6553] #6552

labelsurb = np.concatenate([labelsurb,labelsurb]).astype(int)
best500=[]
for column in ranked:
    best500.append(urban[:,column-1])
best500 = (np.asarray(best500).astype(str)).T
best500 =  np.column_stack((best500,labelsurb))
np.savetxt('urban_AguementedNoise_LLDs.csv',labelsurb,delimiter=",",fmt="%s")
del urban