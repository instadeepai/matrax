# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Game reference:
# -----------------
# [1] Rapoport, A., and M. Guyer. 1966. “A Taxonomy of 2 × 2 Games.” General Systems:
# Yearbook of the Society for General Systems Research 11:203–214.
# [2] Albrecht SV, Ramamoorthy S. Comparative Evaluation of Multiagent Learning Algorithms
# in a Diverse Set of Ad Hoc Team Problems. arXiv preprint arXiv:1907.09189. 2019 Jul 22.
# https://arxiv.org/pdf/1907.09189.pdf

from matrax.games.utils import convert_payoff_vector_to_matrix

# 1 (7)
c1 = [3, 3, 4, 2, 2, 4, 1, 1]
# 2 (8)
c2 = [3, 3, 4, 2, 1, 4, 2, 1]
# 3 (9)
c3 = [3, 3, 4, 1, 1, 4, 2, 2]
# 4 (10)
c4 = [2, 3, 4, 2, 1, 4, 3, 1]
# 5 (11)
c5 = [2, 3, 4, 1, 1, 4, 3, 2]
# 6 (12)
c6 = [2, 2, 4, 1, 1, 4, 3, 3]
# 7 (13)
c7 = [3, 4, 4, 2, 2, 3, 1, 1]
# 8 (14)
c8 = [3, 4, 4, 2, 1, 3, 2, 1]
# 9 (15)
c9 = [3, 4, 4, 1, 2, 3, 1, 2]
# 10 (16)
c10 = [3, 4, 4, 1, 1, 3, 2, 2]
# 11 (17)
c11 = [2, 4, 4, 2, 1, 3, 3, 1]
# 12 (18)
c12 = [2, 4, 4, 1, 1, 3, 3, 2]
# 13 (19)
c13 = [3, 4, 4, 3, 1, 2, 2, 1]
# 14 (20)
c14 = [3, 4, 4, 3, 2, 2, 1, 1]
# 15 (21)
c15 = [2, 4, 4, 3, 1, 2, 3, 1]
# 16 (31)
c16 = [3, 4, 2, 2, 1, 3, 4, 1]
# 17 (32)
c17 = [3, 4, 2, 1, 1, 3, 4, 2]
# 18 (33)
c18 = [3, 4, 1, 2, 2, 3, 4, 1]
# 19 (34)
c19 = [3, 4, 1, 1, 2, 3, 4, 2]
# 20 (35)
c20 = [2, 4, 3, 2, 1, 3, 4, 1]
# 21 (36)
c21 = [2, 4, 3, 1, 1, 3, 4, 2]
# 22 (37)
c22 = [3, 4, 2, 3, 1, 2, 4, 1]
# 23 (38)
c23 = [3, 4, 1, 3, 2, 2, 4, 1]
# 24 (39)
c24 = [2, 4, 3, 3, 1, 2, 4, 1]
# 25 (40)
c25 = [3, 4, 4, 1, 2, 2, 1, 3]
# 26 (41)
c26 = [3, 4, 4, 1, 1, 2, 2, 3]
# 27 (42)
c27 = [3, 3, 4, 1, 2, 2, 1, 4]
# 28 (43)
c28 = [3, 3, 4, 1, 1, 2, 2, 4]
# 29 (44)
c29 = [2, 4, 4, 1, 1, 2, 3, 3]
# 30 (45)
c30 = [3, 2, 4, 1, 2, 3, 1, 4]
# 31 (46)
c31 = [3, 2, 4, 1, 1, 3, 2, 4]
# 32 (47)
c32 = [2, 3, 4, 1, 1, 2, 3, 4]
# 33 (48)
c33 = [2, 2, 4, 1, 1, 3, 3, 4]
# 34 (49)
c34 = [3, 4, 4, 3, 2, 1, 1, 2]
# 35 (50)
c35 = [3, 4, 4, 3, 1, 1, 2, 2]
# 36 (51)
c36 = [3, 4, 4, 2, 2, 1, 1, 3]
# 37 (52)
c37 = [3, 4, 4, 2, 1, 1, 2, 3]
# 38 (53)
c38 = [3, 3, 4, 2, 2, 1, 1, 4]
# 39 (54)
c39 = [3, 3, 4, 2, 1, 1, 2, 4]
# 40 (55)
c40 = [2, 4, 4, 3, 1, 1, 3, 2]
# 41 (56)
c41 = [2, 4, 4, 2, 1, 1, 3, 3]
# 42 (57)
c42 = [2, 3, 4, 2, 1, 1, 3, 4]
# 43 (64)
c43 = [3, 4, 2, 1, 1, 2, 4, 3]
# 44 (65)
c44 = [2, 4, 3, 1, 1, 2, 4, 3]
# 45 (66)
c45 = [3, 3, 2, 4, 4, 2, 1, 1]
# 46 (67)
c46 = [2, 3, 3, 4, 4, 2, 1, 1]
# 47 (68)
c47 = [2, 2, 3, 4, 4, 3, 1, 1]
# 48 (69)
c48 = [2, 2, 4, 3, 3, 4, 1, 1]
# 49 (70)
c49 = [3, 4, 2, 1, 4, 2, 1, 3]
# 50 (71)
c50 = [3, 3, 2, 1, 4, 2, 1, 4]
# 51 (72)
c51 = [3, 2, 2, 1, 4, 3, 1, 4]
# 52 (73)
c52 = [2, 4, 4, 1, 3, 2, 1, 3]
# 53 (74)
c53 = [2, 4, 3, 1, 4, 2, 1, 3]
# 54 (75)
c54 = [2, 3, 4, 1, 3, 2, 1, 4]
# 55 (76)
c55 = [2, 3, 3, 1, 4, 2, 1, 4]
# 56 (77)
c56 = [2, 2, 4, 1, 3, 3, 1, 4]
# 57 (78)
c57 = [2, 2, 3, 1, 4, 3, 1, 4]

games_list = [
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    c8,
    c9,
    c10,
    c11,
    c12,
    c13,
    c14,
    c15,
    c16,
    c17,
    c18,
    c19,
    c20,
    c21,
    c22,
    c23,
    c24,
    c25,
    c26,
    c27,
    c28,
    c29,
    c30,
    c31,
    c32,
    c33,
    c34,
    c35,
    c36,
    c37,
    c38,
    c39,
    c40,
    c41,
    c42,
    c43,
    c44,
    c45,
    c46,
    c47,
    c48,
    c49,
    c50,
    c51,
    c52,
    c53,
    c54,
    c55,
    c56,
    c57,
]

conflict_games = {}

for _id, game in enumerate(games_list):
    conflict_games[f"{_id}"] = convert_payoff_vector_to_matrix(game)
