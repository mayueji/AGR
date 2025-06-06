/*
Copyright (c) 2018, Wenjia Lu, Zuoqiang Shi, Jian Sun and Bin Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of Tsinghua University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#include "MarchingCubes.h"
#include "Geometry.h"
// 8个格点与isovalue的关系对应了256（2^8）种情况
const int MarchingCubes::edgeMask[1<<Cube::CORNERS]={
	0,  273,  545,  816, 2082, 2355, 2563, 2834,
	1042, 1283, 1587, 1826, 3120, 3361, 3601, 3840,
	324,   85,  869,  628, 2406, 2167, 2887, 2646,
	1366, 1095, 1911, 1638, 3444, 3173, 3925, 3652,
	644,  917,  165,  436, 2726, 2999, 2183, 2454,
	1686, 1927, 1207, 1446, 3764, 4005, 3221, 3460,
	960,  721,  481,  240, 3042, 2803, 2499, 2258,
	2002, 1731, 1523, 1250, 4080, 3809, 3537, 3264,
	2184, 2457, 2729, 3000,  170,  443,  651,  922,
	3226, 3467, 3771, 4010, 1208, 1449, 1689, 1928,
	2508, 2269, 3053, 2812,  494,  255,  975,  734,
	3550, 3279, 4095, 3822, 1532, 1261, 2013, 1740,
	2572, 2845, 2093, 2364,  558,  831,   15,  286,
	3614, 3855, 3135, 3374, 1596, 1837, 1053, 1292,
	2888, 2649, 2409, 2168,  874,  635,  331,   90,
	3930, 3659, 3451, 3178, 1912, 1641, 1369, 1096,
	1096, 1369, 1641, 1912, 3178, 3451, 3659, 3930,
	90,  331,  635,  874, 2168, 2409, 2649, 2888,
	1292, 1053, 1837, 1596, 3374, 3135, 3855, 3614,
	286,   15,  831,  558, 2364, 2093, 2845, 2572,
	1740, 2013, 1261, 1532, 3822, 4095, 3279, 3550,
	734,  975,  255,  494, 2812, 3053, 2269, 2508,
	1928, 1689, 1449, 1208, 4010, 3771, 3467, 3226,
	922,  651,  443,  170, 3000, 2729, 2457, 2184,
	3264, 3537, 3809, 4080, 1250, 1523, 1731, 2002,
	2258, 2499, 2803, 3042,  240,  481,  721,  960,
	3460, 3221, 4005, 3764, 1446, 1207, 1927, 1686,
	2454, 2183, 2999, 2726,  436,  165,  917,  644,
	3652, 3925, 3173, 3444, 1638, 1911, 1095, 1366,
	2646, 2887, 2167, 2406,  628,  869,   85,  324,
	3840, 3601, 3361, 3120, 1826, 1587, 1283, 1042,
	2834, 2563, 2355, 2082,  816,  545,  273,    0
};
const int MarchingCubes::triangles[1<<Cube::CORNERS][MAX_TRIANGLES*3+1] = {
	{  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   0,   4,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   5,   0,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   8,   9,   5,   8,   5,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   1,   5,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   0,   4,   8,   1,   5,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   9,  11,   1,   9,   1,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   8,   9,  11,   8,  11,   1,   8,   1,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   4,   1,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  10,   8,   0,  10,   0,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   5,   0,   9,   4,   1,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  10,   8,   9,  10,   9,   5,  10,   5,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  11,  10,   4,  11,   4,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  11,  10,   8,  11,   8,   0,  11,   0,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   9,  11,  10,   9,  10,   4,   9,   4,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   8,   9,  11,   8,  11,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   8,   6,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   6,   2,   0,   4,   6,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   6,   2,   8,   5,   0,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   5,   4,   6,   9,   5,   6,   2,   9,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   1,   5,  11,   8,   6,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   1,   5,  11,   6,   2,   0,   4,   6,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   6,   2,   8,   9,  11,   1,   9,   1,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   9,  11,   2,   2,  11,   1,   2,   1,   6,   6,   1,   4,  -1,  -1,  -1,  -1},
	{   1,  10,   4,   2,   8,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   2,   0,   1,   6,   2,   1,  10,   6,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   5,   0,   9,   4,   1,  10,   8,   6,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   5,   2,   9,   5,   6,   2,   5,   1,   6,   1,  10,   6,  -1,  -1,  -1,  -1},
	{   2,   8,   6,   4,   5,  11,   4,  11,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   5,   2,   0,   6,   2,   5,  11,   6,   5,  10,   6,  11,  -1,  -1,  -1,  -1},
	{   9,  11,  10,   9,  10,   4,   9,   4,   0,   8,   6,   2,  -1,  -1,  -1,  -1},
	{   9,  11,   2,   2,  11,   6,  10,   6,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   9,   2,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   7,   9,   2,   4,   8,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   0,   2,   7,   0,   7,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   7,   5,   4,   2,   7,   4,   8,   2,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   7,   9,   2,   5,  11,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   1,   5,  11,   0,   4,   8,   9,   2,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   1,   0,   2,   1,   2,   7,   1,   7,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   1,   7,  11,   1,   2,   7,   1,   4,   2,   4,   8,   2,  -1,  -1,  -1,  -1},
	{   4,   1,  10,   9,   2,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   7,   9,   2,   0,   1,  10,   0,  10,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   4,   1,  10,   2,   7,   5,   0,   2,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   2,  10,   8,   1,  10,   2,   7,   1,   2,   5,   1,   7,  -1,  -1,  -1,  -1},
	{   7,   9,   2,  10,   4,   5,  11,  10,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  11,  10,   8,  11,   8,   0,  11,   0,   5,   9,   2,   7,  -1,  -1,  -1,  -1},
	{  11,  10,   7,   7,  10,   4,   7,   4,   2,   2,   4,   0,  -1,  -1,  -1,  -1},
	{  11,  10,   7,   7,  10,   2,   8,   2,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   7,   9,   8,   6,   7,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   4,   6,   7,   0,   4,   7,   9,   0,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   6,   7,   5,   8,   6,   5,   0,   8,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   4,   6,   7,   5,   4,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   5,  11,   1,   8,   6,   7,   9,   8,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   4,   6,   7,   0,   4,   7,   9,   0,   7,  11,   1,   5,  -1,  -1,  -1,  -1},
	{   8,   1,   0,  11,   1,   8,   6,  11,   8,   7,  11,   6,  -1,  -1,  -1,  -1},
	{  11,   6,   7,   1,   6,  11,   6,   1,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   1,  10,   4,   6,   7,   9,   6,   9,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   0,   1,   9,   9,   1,  10,   9,  10,   7,   7,  10,   6,  -1,  -1,  -1,  -1},
	{   6,   7,   5,   8,   6,   5,   0,   8,   5,   1,  10,   4,  -1,  -1,  -1,  -1},
	{   1,   7,   5,  10,   7,   1,   7,  10,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  11,  10,   4,  11,   4,   5,   7,   9,   8,   6,   7,   8,  -1,  -1,  -1,  -1},
	{   0,   6,   9,   9,   6,   7,   6,   0,   5,   5,  11,  10,   5,  10,   6,  -1},
	{   8,   7,   0,   6,   7,   8,   4,   0,   7,  11,  10,   4,   7,  11,   4,  -1},
	{  11,  10,   6,  11,   6,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  11,   7,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   0,   4,   8,  11,   7,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   9,   5,   0,  11,   7,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  11,   7,   3,   4,   8,   9,   5,   4,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   3,   1,   5,   3,   5,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   0,   4,   8,   7,   3,   1,   5,   7,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   3,   1,   0,   3,   0,   9,   3,   9,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   7,   8,   9,   4,   8,   7,   3,   4,   7,   1,   4,   3,  -1,  -1,  -1,  -1},
	{   1,  10,   4,   3,  11,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   3,  11,   7,   8,   0,   1,  10,   8,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   4,   1,  10,   5,   0,   9,  11,   7,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  10,   8,   9,  10,   9,   5,  10,   5,   1,  11,   7,   3,  -1,  -1,  -1,  -1},
	{   4,   5,   7,   4,   7,   3,   4,   3,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  10,   8,   3,   3,   8,   0,   3,   0,   7,   7,   0,   5,  -1,  -1,  -1,  -1},
	{   4,   3,  10,   4,   7,   3,   4,   0,   7,   0,   9,   7,  -1,  -1,  -1,  -1},
	{  10,   8,   3,   3,   8,   7,   9,   7,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  11,   7,   3,   8,   6,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  11,   7,   3,   2,   0,   4,   2,   4,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  11,   7,   3,   8,   6,   2,   5,   0,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   5,   4,   6,   9,   5,   6,   2,   9,   6,   3,  11,   7,  -1,  -1,  -1,  -1},
	{   8,   6,   2,   3,   1,   5,   3,   5,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   3,   1,   5,   3,   5,   7,   6,   2,   0,   4,   6,   0,  -1,  -1,  -1,  -1},
	{   3,   1,   0,   3,   0,   9,   3,   9,   7,   2,   8,   6,  -1,  -1,  -1,  -1},
	{   9,   4,   2,   2,   4,   6,   4,   9,   7,   7,   3,   1,   7,   1,   4,  -1},
	{   8,   6,   2,  11,   7,   3,   4,   1,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   2,   0,   1,   6,   2,   1,  10,   6,   1,  11,   7,   3,  -1,  -1,  -1,  -1},
	{   5,   0,   9,   4,   1,  10,   8,   6,   2,  11,   7,   3,  -1,  -1,  -1,  -1},
	{  11,   7,   3,   5,   2,   9,   5,   6,   2,   5,   1,   6,   1,  10,   6,  -1},
	{   4,   5,   7,   4,   7,   3,   4,   3,  10,   6,   2,   8,  -1,  -1,  -1,  -1},
	{  10,   5,   3,   3,   5,   7,   5,  10,   6,   6,   2,   0,   6,   0,   5,  -1},
	{   8,   6,   2,   4,   3,  10,   4,   7,   3,   4,   0,   7,   0,   9,   7,  -1},
	{   9,   7,  10,  10,   7,   3,  10,   6,   9,   6,   2,   9,  -1,  -1,  -1,  -1},
	{   3,  11,   9,   2,   3,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   4,   8,   0,   2,   3,  11,   2,  11,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   0,   2,   3,   0,   3,  11,   0,  11,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   2,   3,   8,   8,   3,  11,   8,  11,   4,   4,  11,   5,  -1,  -1,  -1,  -1},
	{   2,   3,   1,   2,   1,   5,   2,   5,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   2,   3,   1,   2,   1,   5,   2,   5,   9,   0,   4,   8,  -1,  -1,  -1,  -1},
	{   0,   2,   3,   0,   3,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   2,   3,   8,   8,   3,   4,   1,   4,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   1,  10,   4,   9,   2,   3,  11,   9,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  10,   8,   0,  10,   0,   1,   3,  11,   9,   2,   3,   9,  -1,  -1,  -1,  -1},
	{   0,   2,   3,   0,   3,  11,   0,  11,   5,   1,  10,   4,  -1,  -1,  -1,  -1},
	{   5,   2,  11,  11,   2,   3,   2,   5,   1,   1,  10,   8,   1,   8,   2,  -1},
	{  10,   2,   3,   9,   2,  10,   4,   9,  10,   5,   9,   4,  -1,  -1,  -1,  -1},
	{   5,  10,   0,   0,  10,   8,  10,   5,   9,   9,   2,   3,   9,   3,  10,  -1},
	{   0,   2,   4,   4,   2,  10,   3,  10,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  10,   8,   2,  10,   2,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  11,   9,   8,   3,  11,   8,   6,   3,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   0,  11,   9,   3,  11,   0,   4,   3,   0,   6,   3,   4,  -1,  -1,  -1,  -1},
	{  11,   5,   3,   5,   0,   3,   0,   6,   3,   0,   8,   6,  -1,  -1,  -1,  -1},
	{   3,   4,   6,  11,   4,   3,   4,  11,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   3,   1,   6,   6,   1,   5,   6,   5,   8,   8,   5,   9,  -1,  -1,  -1,  -1},
	{   0,   6,   9,   4,   6,   0,   5,   9,   6,   3,   1,   5,   6,   3,   5,  -1},
	{   3,   1,   6,   6,   1,   8,   0,   8,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   3,   1,   4,   3,   4,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  11,   9,   8,   3,  11,   8,   6,   3,   8,   4,   1,  10,  -1,  -1,  -1,  -1},
	{   3,   9,   6,  11,   9,   3,  10,   6,   9,   0,   1,  10,   9,   0,  10,  -1},
	{   4,   1,  10,  11,   5,   3,   5,   0,   3,   0,   6,   3,   0,   8,   6,  -1},
	{   5,  10,   6,   1,  10,   5,   6,  11,   5,   6,   3,  11,  -1,  -1,  -1,  -1},
	{  10,   5,   3,   4,   5,  10,   6,   3,   5,   9,   8,   6,   5,   9,   6,  -1},
	{   6,   3,  10,   9,   0,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   3,  10,   0,   0,  10,   4,   0,   8,   3,   8,   6,   3,  -1,  -1,  -1,  -1},
	{   6,   3,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  10,   3,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   3,   6,  10,   0,   4,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   5,   0,   9,  10,   3,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   3,   6,  10,   8,   9,   5,   8,   5,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  11,   1,   5,  10,   3,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   0,   4,   8,   1,   5,  11,  10,   3,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  10,   3,   6,   0,   9,  11,   1,   0,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   8,   9,  11,   8,  11,   1,   8,   1,   4,  10,   3,   6,  -1,  -1,  -1,  -1},
	{   4,   1,   3,   6,   4,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   0,   1,   3,   8,   0,   3,   6,   8,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   5,   0,   9,   3,   6,   4,   1,   3,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   8,   9,   6,   6,   9,   5,   6,   5,   3,   3,   5,   1,  -1,  -1,  -1,  -1},
	{   6,   4,   5,   6,   5,  11,   6,  11,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   0,   6,   8,   0,   3,   6,   0,   5,   3,   5,  11,   3,  -1,  -1,  -1,  -1},
	{   3,   9,  11,   0,   9,   3,   6,   0,   3,   4,   0,   6,  -1,  -1,  -1,  -1},
	{   8,   9,   6,   6,   9,   3,  11,   3,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   2,   8,  10,   3,   2,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   3,   2,   0,  10,   3,   0,   4,  10,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   5,   0,   9,   8,  10,   3,   8,   3,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   9,   3,   2,  10,   3,   9,   5,  10,   9,   4,  10,   5,  -1,  -1,  -1,  -1},
	{  11,   1,   5,   2,   8,  10,   3,   2,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   3,   2,   0,  10,   3,   0,   4,  10,   0,   5,  11,   1,  -1,  -1,  -1,  -1},
	{   9,  11,   1,   9,   1,   0,   2,   8,  10,   3,   2,  10,  -1,  -1,  -1,  -1},
	{  10,   2,   4,   3,   2,  10,   1,   4,   2,   9,  11,   1,   2,   9,   1,  -1},
	{   1,   3,   2,   4,   1,   2,   8,   4,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   0,   1,   3,   2,   0,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   1,   3,   2,   4,   1,   2,   8,   4,   2,   9,   5,   0,  -1,  -1,  -1,  -1},
	{   9,   3,   2,   5,   3,   9,   3,   5,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   3,   2,  11,  11,   2,   8,  11,   8,   5,   5,   8,   4,  -1,  -1,  -1,  -1},
	{   5,   2,   0,  11,   2,   5,   2,  11,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   4,   3,   8,   8,   3,   2,   3,   4,   0,   0,   9,  11,   0,  11,   3,  -1},
	{   9,  11,   3,   9,   3,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  10,   3,   6,   9,   2,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   9,   2,   7,  10,   3,   6,   0,   4,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  10,   3,   6,   7,   5,   0,   7,   0,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   7,   5,   4,   2,   7,   4,   8,   2,   4,  10,   3,   6,  -1,  -1,  -1,  -1},
	{  10,   3,   6,   9,   2,   7,   1,   5,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  10,   3,   6,   9,   2,   7,   1,   5,  11,   0,   4,   8,  -1,  -1,  -1,  -1},
	{   1,   0,   2,   1,   2,   7,   1,   7,  11,   3,   6,  10,  -1,  -1,  -1,  -1},
	{  10,   3,   6,   1,   7,  11,   1,   2,   7,   1,   4,   2,   4,   8,   2,  -1},
	{   9,   2,   7,   6,   4,   1,   6,   1,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   0,   1,   3,   8,   0,   3,   6,   8,   3,   7,   9,   2,  -1,  -1,  -1,  -1},
	{   0,   2,   7,   0,   7,   5,   4,   1,   3,   6,   4,   3,  -1,  -1,  -1,  -1},
	{   2,   5,   8,   7,   5,   2,   6,   8,   5,   1,   3,   6,   5,   1,   6,  -1},
	{   6,   4,   5,   6,   5,  11,   6,  11,   3,   7,   9,   2,  -1,  -1,  -1,  -1},
	{   9,   2,   7,   0,   6,   8,   0,   3,   6,   0,   5,   3,   5,  11,   3,  -1},
	{   3,   4,  11,   6,   4,   3,   7,  11,   4,   0,   2,   7,   4,   0,   7,  -1},
	{  11,   3,   8,   8,   3,   6,   8,   2,  11,   2,   7,  11,  -1,  -1,  -1,  -1},
	{   9,   8,  10,   7,   9,  10,   3,   7,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   9,   0,   7,   0,   4,   7,   4,   3,   7,   4,  10,   3,  -1,  -1,  -1,  -1},
	{   8,  10,   0,   0,  10,   3,   0,   3,   5,   5,   3,   7,  -1,  -1,  -1,  -1},
	{  10,   5,   4,   3,   5,  10,   5,   3,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   9,   8,  10,   7,   9,  10,   3,   7,  10,   1,   5,  11,  -1,  -1,  -1,  -1},
	{   1,   5,  11,   9,   0,   7,   0,   4,   7,   4,   3,   7,   4,  10,   3,  -1},
	{  11,   0,   7,   1,   0,  11,   3,   7,   0,   8,  10,   3,   0,   8,   3,  -1},
	{   7,   1,   4,  11,   1,   7,   4,   3,   7,   4,  10,   3,  -1,  -1,  -1,  -1},
	{   4,   9,   8,   7,   9,   4,   1,   7,   4,   3,   7,   1,  -1,  -1,  -1,  -1},
	{   7,   1,   3,   9,   1,   7,   1,   9,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   8,   7,   0,   0,   7,   5,   7,   8,   4,   4,   1,   3,   4,   3,   7,  -1},
	{   5,   1,   3,   7,   5,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   3,   4,  11,  11,   4,   5,   4,   3,   7,   7,   9,   8,   7,   8,   4,  -1},
	{   3,   9,   0,   7,   9,   3,   0,  11,   3,   0,   5,  11,  -1,  -1,  -1,  -1},
	{   3,   7,  11,   8,   4,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   3,   7,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   6,  10,  11,   7,   6,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   0,   4,   8,  10,  11,   7,  10,   7,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   9,   5,   0,   6,  10,  11,   7,   6,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   8,   9,   5,   8,   5,   4,   6,  10,  11,   7,   6,  11,  -1,  -1,  -1,  -1},
	{   5,   7,   6,   5,   6,  10,   5,  10,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   5,   7,   6,   5,   6,  10,   5,  10,   1,   4,   8,   0,  -1,  -1,  -1,  -1},
	{   1,   0,  10,  10,   0,   9,  10,   9,   6,   6,   9,   7,  -1,  -1,  -1,  -1},
	{   1,   7,  10,  10,   7,   6,   7,   1,   4,   4,   8,   9,   4,   9,   7,  -1},
	{   7,   6,   4,   7,   4,   1,   7,   1,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  11,   0,   1,   8,   0,  11,   7,   8,  11,   6,   8,   7,  -1,  -1,  -1,  -1},
	{   7,   6,   4,   7,   4,   1,   7,   1,  11,   5,   0,   9,  -1,  -1,  -1,  -1},
	{  11,   6,   1,   7,   6,  11,   5,   1,   6,   8,   9,   5,   6,   8,   5,  -1},
	{   4,   5,   7,   4,   7,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   5,   7,   0,   0,   7,   8,   6,   8,   7,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   7,   6,   9,   9,   6,   0,   4,   0,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   8,   9,   7,   8,   7,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   8,  10,  11,   2,   8,  11,   7,   2,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  10,  11,   4,   4,  11,   7,   4,   7,   0,   0,   7,   2,  -1,  -1,  -1,  -1},
	{   8,  10,  11,   2,   8,  11,   7,   2,  11,   5,   0,   9,  -1,  -1,  -1,  -1},
	{   9,   4,   2,   5,   4,   9,   7,   2,   4,  10,  11,   7,   4,  10,   7,  -1},
	{   1,   8,  10,   2,   8,   1,   5,   2,   1,   7,   2,   5,  -1,  -1,  -1,  -1},
	{   1,   7,  10,   5,   7,   1,   4,  10,   7,   2,   0,   4,   7,   2,   4,  -1},
	{   7,   1,   9,   9,   1,   0,   1,   7,   2,   2,   8,  10,   2,  10,   1,  -1},
	{   7,   2,   9,  10,   1,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   8,   4,   2,   4,   1,   2,   1,   7,   2,   1,  11,   7,  -1,  -1,  -1,  -1},
	{  11,   0,   1,   7,   0,  11,   0,   7,   2,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   5,   0,   9,   8,   4,   2,   4,   1,   2,   1,   7,   2,   1,  11,   7,  -1},
	{   2,   5,   1,   9,   5,   2,   1,   7,   2,   1,  11,   7,  -1,  -1,  -1,  -1},
	{   4,   5,   8,   8,   5,   2,   7,   2,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   7,   2,   0,   5,   7,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   7,   2,   4,   4,   2,   8,   4,   0,   7,   0,   9,   7,  -1,  -1,  -1,  -1},
	{   7,   2,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  10,  11,   9,   6,  10,   9,   2,   6,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  10,  11,   9,   6,  10,   9,   2,   6,   9,   0,   4,   8,  -1,  -1,  -1,  -1},
	{   5,  10,  11,   6,  10,   5,   0,   6,   5,   2,   6,   0,  -1,  -1,  -1,  -1},
	{   2,   5,   8,   8,   5,   4,   5,   2,   6,   6,  10,  11,   6,  11,   5,  -1},
	{  10,   1,   6,   1,   5,   6,   5,   2,   6,   5,   9,   2,  -1,  -1,  -1,  -1},
	{   0,   4,   8,  10,   1,   6,   1,   5,   6,   5,   2,   6,   5,   9,   2,  -1},
	{   1,   0,  10,  10,   0,   6,   2,   6,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   2,   6,   1,   1,   6,  10,   1,   4,   2,   4,   8,   2,  -1,  -1,  -1,  -1},
	{  11,   9,   1,   1,   9,   2,   1,   2,   4,   4,   2,   6,  -1,  -1,  -1,  -1},
	{   8,   1,   6,   0,   1,   8,   2,   6,   1,  11,   9,   2,   1,  11,   2,  -1},
	{  11,   6,   1,   1,   6,   4,   6,  11,   5,   5,   0,   2,   5,   2,   6,  -1},
	{   2,   6,   8,  11,   5,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   6,   4,   2,   2,   4,   9,   5,   9,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   5,   9,   6,   6,   9,   2,   6,   8,   5,   8,   0,   5,  -1,  -1,  -1,  -1},
	{   0,   2,   6,   0,   6,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   2,   6,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   8,  10,  11,   9,   8,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   0,  11,   9,   4,  11,   0,  11,   4,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   5,  10,  11,   0,  10,   5,  10,   0,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   4,  10,  11,   5,   4,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   1,   8,  10,   5,   8,   1,   8,   5,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   9,   4,  10,   0,   4,   9,  10,   5,   9,  10,   1,   5,  -1,  -1,  -1,  -1},
	{   0,   8,  10,   1,   0,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  10,   1,   4,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   4,   9,   8,   1,   9,   4,   9,   1,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   1,  11,   9,   0,   1,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  11,   0,   8,   5,   0,  11,   8,   1,  11,   8,   4,   1,  -1,  -1,  -1,  -1},
	{  11,   5,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   5,   9,   8,   4,   5,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   9,   0,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{   8,   4,   0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	{  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1}
};
const int MarchingCubes::cornerMap[Cube::CORNERS]={0,1,3,2,4,5,7,6};
double MarchingCubes::vertexList[Cube::EDGES][3];

int MarchingCubes::GetIndex(const double v[Cube::CORNERS],const double& iso){
	int idx=0;
	if (v[Cube::CornerIndex(0,0,0)] < iso) idx |=   1;
	if (v[Cube::CornerIndex(1,0,0)] < iso) idx |=   2;
	if (v[Cube::CornerIndex(1,1,0)] < iso) idx |=   4;
	if (v[Cube::CornerIndex(0,1,0)] < iso) idx |=   8;
	if (v[Cube::CornerIndex(0,0,1)] < iso) idx |=  16;
	if (v[Cube::CornerIndex(1,0,1)] < iso) idx |=  32;
	if (v[Cube::CornerIndex(1,1,1)] < iso) idx |=  64;
	if (v[Cube::CornerIndex(0,1,1)] < iso) idx |= 128;
	return idx;
}

int MarchingCubes::GetFaceIndex(const double values[Cube::CORNERS],const double& iso,const int& faceIndex){
	int i,j,x,y,z,idx=0;
	double v[2][2];
	Cube::FactorFaceIndex(faceIndex,x,y,z);
	if		(x<0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=values[Cube::CornerIndex(0,i,j)];}}}
	else if	(x>0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=values[Cube::CornerIndex(1,i,j)];}}}
	else if	(y<0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=values[Cube::CornerIndex(i,0,j)];}}}
	else if	(y>0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=values[Cube::CornerIndex(i,1,j)];}}}
	else if	(z<0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=values[Cube::CornerIndex(i,j,0)];}}}
	else if	(z>0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=values[Cube::CornerIndex(i,j,1)];}}}
	if (v[0][0] < iso) idx |=   1;
	if (v[1][0] < iso) idx |=   2;
	if (v[1][1] < iso) idx |=   4;
	if (v[0][1] < iso) idx |=   8;
	return idx;
}
int MarchingCubes::IsAmbiguous(const double v[Cube::CORNERS],const double& isoValue,const int& faceIndex){
	int idx=GetFaceIndex(v,isoValue,faceIndex);
	return (idx==5) || (idx==10);
}
int MarchingCubes::HasRoots(const double v[Cube::CORNERS],const double& isoValue,const int& faceIndex){
	int idx=GetFaceIndex(v,isoValue,faceIndex);
	return (idx!=0) && (idx !=15);
}
int MarchingCubes::HasRoots(const double v[Cube::CORNERS],const double& isoValue){
	int idx=GetIndex(v,isoValue);
	if(idx==0 || idx==255){return 0;}
	else{return 1;}
}
int MarchingCubes::HasRoots(const int& mcIndex){
	if(mcIndex==0 || mcIndex==255){return 0;}
	else{return 1;}
}
int MarchingCubes::AddTriangles(const double v[Cube::CORNERS],const double& iso,Triangle* isoTriangles){
	int idx,ntriang=0;
	Triangle tri;

	idx=GetIndex(v,iso);

	/* Cube is entirely in/out of the surface */
	if (!edgeMask[idx]) return 0;

	/* Find the vertices where the surface intersects the cube */
	int i,j,ii=1;
	for(i=0;i<12;i++){
		if(edgeMask[idx] & ii){SetVertex(i,v,iso);}
		ii<<=1;
	}
	/* Create the triangle */
	for (i=0;triangles[idx][i]!=-1;i+=3) {
		for(j=0;j<3;j++){
			tri.p[0][j]=vertexList[triangles[idx][i+0]][j];
			tri.p[1][j]=vertexList[triangles[idx][i+1]][j];
			tri.p[2][j]=vertexList[triangles[idx][i+2]][j];
		}
		isoTriangles[ntriang++]=tri;
	}
	return ntriang;
}

int MarchingCubes::AddTriangleIndices(const double v[Cube::CORNERS],const double& iso,int* isoIndices){
	int idx,ntriang=0;

	idx=GetIndex(v,iso);

	/* Cube is entirely in/out of the surface */
	if (!edgeMask[idx]) return 0;

	/* Create the triangle */
	for(int i=0;triangles[idx][i]!=-1;i+=3){
		for(int j=0;j<3;j++){isoIndices[i+j]=triangles[idx][i+j];}
		ntriang++;
	}
	return ntriang;
}

void MarchingCubes::SetVertex(const int& e,const double values[Cube::CORNERS],const double& iso){
	double t;
	switch(e){
	case 0:
		t=Interpolate(values[Cube::CornerIndex(0,0,0)]-iso,values[Cube::CornerIndex(1,0,0)]-iso);
		vertexList[e][0]=t;			vertexList[e][1]=0.0;		vertexList[e][2]=0.0;
		break;
	case 1:
		t=Interpolate(values[Cube::CornerIndex(1,0,0)]-iso,values[Cube::CornerIndex(1,1,0)]-iso);
		vertexList[e][0]=1.0;		vertexList[e][1]=t;			vertexList[e][2]=0.0;
		break;
	case 2:
		t=Interpolate(values[Cube::CornerIndex(1,1,0)]-iso,values[Cube::CornerIndex(0,1,0)]-iso);
		vertexList[e][0]=(1.0-t);	vertexList[e][1]=1.0;		vertexList[e][2]=0.0;
		break;
	case 3:
		t=Interpolate(values[Cube::CornerIndex(0,1,0)]-iso,values[Cube::CornerIndex(0,0,0)]-iso);
		vertexList[e][0]=0.0;		vertexList[e][1]=(1.0-t);	vertexList[e][2]=0.0;
		break;
	case 4:
		t=Interpolate(values[Cube::CornerIndex(0,0,1)]-iso,values[Cube::CornerIndex(1,0,1)]-iso);
		vertexList[e][0]=t;			vertexList[e][1]=0.0;		vertexList[e][2]=1.0;
		break;
	case 5:
		t=Interpolate(values[Cube::CornerIndex(1,0,1)]-iso,values[Cube::CornerIndex(1,1,1)]-iso);
		vertexList[e][0]=1.0;		vertexList[e][1]=t;			vertexList[e][2]=1.0;
		break;
	case 6:
		t=Interpolate(values[Cube::CornerIndex(1,1,1)]-iso,values[Cube::CornerIndex(0,1,1)]-iso);
		vertexList[e][0]=(1.0-t);	vertexList[e][1]=1.0;		vertexList[e][2]=1.0;
		break;
	case 7:
		t=Interpolate(values[Cube::CornerIndex(0,1,1)]-iso,values[Cube::CornerIndex(0,0,1)]-iso);
		vertexList[e][0]=0.0;		vertexList[e][1]=(1.0-t);	vertexList[e][2]=1.0;
		break;
	case 8:
		t=Interpolate(values[Cube::CornerIndex(0,0,0)]-iso,values[Cube::CornerIndex(0,0,1)]-iso);
		vertexList[e][0]=0.0;		vertexList[e][1]=0.0;		vertexList[e][2]=t;
		break;
	case 9:
		t=Interpolate(values[Cube::CornerIndex(1,0,0)]-iso,values[Cube::CornerIndex(1,0,1)]-iso);
		vertexList[e][0]=1.0;		vertexList[e][1]=0.0;		vertexList[e][2]=t;
		break;
	case 10:
		t=Interpolate(values[Cube::CornerIndex(1,1,0)]-iso,values[Cube::CornerIndex(1,1,1)]-iso);
		vertexList[e][0]=1.0;		vertexList[e][1]=1.0;		vertexList[e][2]=t;
		break;
	case 11:
		t=Interpolate(values[Cube::CornerIndex(0,1,0)]-iso,values[Cube::CornerIndex(0,1,1)]-iso);
		vertexList[e][0]=0.0;		vertexList[e][1]=1.0;		vertexList[e][2]=t;
		break;
	};
}
double MarchingCubes::Interpolate(const double& v1,const double& v2){return v1/(v1-v2);}

///////////////////////////////////
int MarchingCubes::GetIndex(const float v[Cube::CORNERS],const float& iso){
	int idx=0;
	if (v[Cube::CornerIndex(0,0,0)] < iso) idx |=   1;
	if (v[Cube::CornerIndex(1,0,0)] < iso) idx |=   2;
	if (v[Cube::CornerIndex(1,1,0)] < iso) idx |=   4;
	if (v[Cube::CornerIndex(0,1,0)] < iso) idx |=   8;
	if (v[Cube::CornerIndex(0,0,1)] < iso) idx |=  16;
	if (v[Cube::CornerIndex(1,0,1)] < iso) idx |=  32;
	if (v[Cube::CornerIndex(1,1,1)] < iso) idx |=  64;
	if (v[Cube::CornerIndex(0,1,1)] < iso) idx |= 128;
	return idx;
}
int MarchingCubes::GetFaceIndex(const float values[Cube::CORNERS],const float& iso,const int& faceIndex){
	int i,j,x,y,z,idx=0;
	double v[2][2];
	Cube::FactorFaceIndex(faceIndex,x,y,z);
	if		(x<0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=values[Cube::CornerIndex(0,i,j)];}}}
	else if	(x>0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=values[Cube::CornerIndex(1,i,j)];}}}
	else if	(y<0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=values[Cube::CornerIndex(i,0,j)];}}}
	else if	(y>0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=values[Cube::CornerIndex(i,1,j)];}}}
	else if	(z<0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=values[Cube::CornerIndex(i,j,0)];}}}
	else if	(z>0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=values[Cube::CornerIndex(i,j,1)];}}}
	if (v[0][0] < iso) idx |=   1;
	if (v[1][0] < iso) idx |=   2;
	if (v[1][1] < iso) idx |=   4;
	if (v[0][1] < iso) idx |=   8;
	return idx;
}
int MarchingCubes::GetFaceIndex(const int& mcIndex,const int& faceIndex){
	int i,j,x,y,z,idx=0;
	int v[2][2];
	Cube::FactorFaceIndex(faceIndex,x,y,z);
	if		(x<0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=mcIndex&(1<<MarchingCubes::cornerMap[Cube::CornerIndex(0,i,j)]);}}}
	else if	(x>0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=mcIndex&(1<<MarchingCubes::cornerMap[Cube::CornerIndex(1,i,j)]);}}}
	else if	(y<0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=mcIndex&(1<<MarchingCubes::cornerMap[Cube::CornerIndex(i,0,j)]);}}}
	else if	(y>0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=mcIndex&(1<<MarchingCubes::cornerMap[Cube::CornerIndex(i,1,j)]);}}}
	else if	(z<0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=mcIndex&(1<<MarchingCubes::cornerMap[Cube::CornerIndex(i,j,1)]);}}}
	else if	(z>0){for(i=0;i<2;i++){for(j=0;j<2;j++){v[i][j]=mcIndex&(1<<MarchingCubes::cornerMap[Cube::CornerIndex(i,j,1)]);}}}
	if (v[0][0]) idx |=   1;
	if (v[1][0]) idx |=   2;
	if (v[1][1]) idx |=   4;
	if (v[0][1]) idx |=   8;
	return idx;
}
int MarchingCubes::IsAmbiguous(const float v[Cube::CORNERS],const float& isoValue,const int& faceIndex){
	int idx=GetFaceIndex(v,isoValue,faceIndex);
	return (idx==5) || (idx==10);
}
int MarchingCubes::IsAmbiguous(const int& mcIndex,const int& faceIndex){
	int idx=GetFaceIndex(mcIndex,faceIndex);
	return (idx==5) || (idx==10);
}
int MarchingCubes::HasRoots(const float v[Cube::CORNERS],const float& isoValue){
	int idx=GetIndex(v,isoValue);
	if(idx==0 || idx==255){return 0;}
	else{return 1;}
}
int MarchingCubes::HasRoots(const float v[Cube::CORNERS],const float& isoValue,const int& faceIndex){
	int idx=GetFaceIndex(v,isoValue,faceIndex);
	return (idx!=0) && (idx!=15);
}
int MarchingCubes::HasFaceRoots(const int& mcIndex,const int& faceIndex){
	int idx=GetFaceIndex(mcIndex,faceIndex);
	return (idx!=0) && (idx!=15);
}

// egdeIndex边上有没有根
int MarchingCubes::HasEdgeRoots(const int& mcIndex,const int& edgeIndex){
	int c1,c2;
	Cube::EdgeCorners(edgeIndex,c1,c2);
	if(	( (mcIndex&(1<<MarchingCubes::cornerMap[c1])) &&  (mcIndex&(1<<MarchingCubes::cornerMap[c2]))) ||
		(!(mcIndex&(1<<MarchingCubes::cornerMap[c1])) && !(mcIndex&(1<<MarchingCubes::cornerMap[c2])))){return 0;}
	else{return 1;}
}
int MarchingCubes::AddTriangles(const float v[Cube::CORNERS],const float& iso,Triangle* isoTriangles){
	int idx,ntriang=0;
	Triangle tri;

	idx=GetIndex(v,iso);

	/* Cube is entirely in/out of the surface */
	if (!edgeMask[idx]) return 0;

	/* Find the vertices where the surface intersects the cube */
	int i,j,ii=1;
	for(i=0;i<12;i++){
		if(edgeMask[idx] & ii){SetVertex(i,v,iso);}
		ii<<=1;
	}
	/* Create the triangle */
	for (i=0;triangles[idx][i]!=-1;i+=3) {
		for(j=0;j<3;j++){
			tri.p[0][j]=vertexList[triangles[idx][i+0]][j];
			tri.p[1][j]=vertexList[triangles[idx][i+1]][j];
			tri.p[2][j]=vertexList[triangles[idx][i+2]][j];
		}
		isoTriangles[ntriang++]=tri;
	}
	return ntriang;
}

int MarchingCubes::AddTriangleIndices(const float v[Cube::CORNERS],const float& iso,int* isoIndices){
	int idx,ntriang=0;
	idx=GetIndex(v,iso);
	/* Cube is entirely in/out of the surface */
	if (!edgeMask[idx]) return 0;
	/* Create the triangle */
	for(int i=0;triangles[idx][i]!=-1;i+=3){
		for(int j=0;j<3;j++){isoIndices[i+j]=triangles[idx][i+j];}
		ntriang++;
	}
	return ntriang;
}
int MarchingCubes::AddTriangleIndices(const int& idx,int* isoIndices){
	int ntriang=0;

	/* Cube is entirely in/out of the surface */
	if (!edgeMask[idx]) return 0;

	/* Create the triangle */
	for(int i=0;triangles[idx][i]!=-1;i+=3){
		for(int j=0;j<3;j++){isoIndices[i+j]=triangles[idx][i+j];}
		ntriang++;
	}
	return ntriang;
}

void MarchingCubes::SetVertex(const int& e,const float values[Cube::CORNERS],const float& iso){
	double t;
	switch(e){
	case 0:
		t=Interpolate(values[Cube::CornerIndex(0,0,0)]-iso,values[Cube::CornerIndex(1,0,0)]-iso);
		vertexList[e][0]=t;			vertexList[e][1]=0.0;		vertexList[e][2]=0.0;
		break;
	case 1:
		t=Interpolate(values[Cube::CornerIndex(1,0,0)]-iso,values[Cube::CornerIndex(1,1,0)]-iso);
		vertexList[e][0]=1.0;		vertexList[e][1]=t;			vertexList[e][2]=0.0;
		break;
	case 2:
		t=Interpolate(values[Cube::CornerIndex(1,1,0)]-iso,values[Cube::CornerIndex(0,1,0)]-iso);
		vertexList[e][0]=(1.0-t);	vertexList[e][1]=1.0;		vertexList[e][2]=0.0;
		break;
	case 3:
		t=Interpolate(values[Cube::CornerIndex(0,1,0)]-iso,values[Cube::CornerIndex(0,0,0)]-iso);
		vertexList[e][0]=0.0;		vertexList[e][1]=(1.0-t);	vertexList[e][2]=0.0;
		break;
	case 4:
		t=Interpolate(values[Cube::CornerIndex(0,0,1)]-iso,values[Cube::CornerIndex(1,0,1)]-iso);
		vertexList[e][0]=t;			vertexList[e][1]=0.0;		vertexList[e][2]=1.0;
		break;
	case 5:
		t=Interpolate(values[Cube::CornerIndex(1,0,1)]-iso,values[Cube::CornerIndex(1,1,1)]-iso);
		vertexList[e][0]=1.0;		vertexList[e][1]=t;			vertexList[e][2]=1.0;
		break;
	case 6:
		t=Interpolate(values[Cube::CornerIndex(1,1,1)]-iso,values[Cube::CornerIndex(0,1,1)]-iso);
		vertexList[e][0]=(1.0-t);	vertexList[e][1]=1.0;		vertexList[e][2]=1.0;
		break;
	case 7:
		t=Interpolate(values[Cube::CornerIndex(0,1,1)]-iso,values[Cube::CornerIndex(0,0,1)]-iso);
		vertexList[e][0]=0.0;		vertexList[e][1]=(1.0-t);	vertexList[e][2]=1.0;
		break;
	case 8:
		t=Interpolate(values[Cube::CornerIndex(0,0,0)]-iso,values[Cube::CornerIndex(0,0,1)]-iso);
		vertexList[e][0]=0.0;		vertexList[e][1]=0.0;		vertexList[e][2]=t;
		break;
	case 9:
		t=Interpolate(values[Cube::CornerIndex(1,0,0)]-iso,values[Cube::CornerIndex(1,0,1)]-iso);
		vertexList[e][0]=1.0;		vertexList[e][1]=0.0;		vertexList[e][2]=t;
		break;
	case 10:
		t=Interpolate(values[Cube::CornerIndex(1,1,0)]-iso,values[Cube::CornerIndex(1,1,1)]-iso);
		vertexList[e][0]=1.0;		vertexList[e][1]=1.0;		vertexList[e][2]=t;
		break;
	case 11:
		t=Interpolate(values[Cube::CornerIndex(0,1,0)]-iso,values[Cube::CornerIndex(0,1,1)]-iso);
		vertexList[e][0]=0.0;		vertexList[e][1]=1.0;		vertexList[e][2]=t;
		break;
	};
}
float MarchingCubes::Interpolate(const float& v1,const float& v2){return v1/(v1-v2);}

