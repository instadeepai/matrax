# Copyright 2023 InstaDeep Ltd. All rights reserved.
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

# 1 (1)
nc1 = [4, 4, 3, 3, 2, 2, 1, 1]
# 2 (2)
nc2 = [4, 4, 3, 3, 1, 2, 2, 1]
# 3 (3)
nc3 = [4, 4, 3, 2, 2, 3, 1, 1]
# 4 (4)
nc4 = [4, 4, 3, 2, 1, 3, 2, 1]
# 5 (5)
nc5 = [4, 4, 3, 1, 1, 3, 2, 2]
# 6 (6)
nc6 = [4, 4, 2, 3, 3, 2, 1, 1]
# 7 (22)
nc7 = [4, 4, 3, 3, 2, 1, 1, 2]
# 8 (23)
nc8 = [4, 4, 3, 3, 1, 1, 2, 2]
# 9 (24)
nc9 = [4, 4, 3, 2, 2, 1, 1, 3]
# 10 (25)
nc10 = [4, 4, 3, 2, 1, 1, 2, 3]
# 11 (26)
nc11 = [4, 4, 2, 3, 3, 1, 1, 2]
# 12 (27)
nc12 = [4, 4, 2, 2, 3, 1, 1, 3]
# 13 (28)
nc13 = [4, 4, 3, 1, 2, 2, 1, 3]
# 14 (29)
nc14 = [4, 4, 3, 1, 1, 2, 2, 3]
# 15 (30)
nc15 = [4, 4, 2, 1, 3, 2, 1, 3]
# 16 (58)
nc16 = [4, 4, 2, 3, 1, 1, 3, 2]
# 17 (59)
nc17 = [4, 4, 2, 2, 1, 1, 3, 3]
# 18 (60)
nc18 = [4, 4, 2, 1, 1, 2, 3, 3]
# 19 (61)
nc19 = [4, 4, 1, 3, 3, 1, 2, 2]
# 20 (62)
nc20 = [4, 4, 1, 2, 3, 1, 2, 3]
# 21 (63)
nc21 = [4, 4, 1, 2, 2, 1, 3, 3]

games_list = [
    nc1,
    nc2,
    nc3,
    nc4,
    nc5,
    nc6,
    nc7,
    nc8,
    nc9,
    nc10,
    nc11,
    nc12,
    nc13,
    nc14,
    nc15,
    nc16,
    nc17,
    nc18,
    nc19,
    nc20,
    nc21,
]

no_conflict_games = {}

for _id, game in enumerate(games_list):
    no_conflict_games[f"{_id}"] = convert_payoff_vector_to_matrix(game)
