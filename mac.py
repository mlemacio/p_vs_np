import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# Input, set of coordinates
coords_list = np.array([(1, 1), (4, 2), (5, 2), (6, 4),
                       (4, 4), (3, 6), (1, 5), (2, 3)])

coords_list = np.array([(-0.0000000400893815,       0.0000000358808126),
                        (-28.8732862244731230,     -0.0000008724121069),
                        (-79.2915791686897506,      21.4033307581457670),
                        (-14.6577381710829471,      43.3895496964974043),
                        (-64.7472605264735108,     -21.8981713360336698),
                        (-29.0584693142401171,      43.2167287683090606),
                        (-72.0785319657452987,      -0.1815834632498404),
                        (-36.0366489745023770,      21.6135482886620949),
                        (-50.4808382862985496,      -7.3744722432402208),
                        (-50.5859026832315024,      21.5881966132975371),
                        (-0.1358203773809326,      28.7292896751977480),
                        (-65.0865638413727368,      36.0624693073746769),
                        (-21.4983260706612533,      -7.3194159498090388),
                        (-57.5687244704708050,      43.2505562436354225),
                        (-43.0700258454450875,     -14.5548396888330487)])

coords_list = np.array([(-0.0000000400893815,       0.0000000358808126),
                        (-28.8732862244731230,     -0.0000008724121069),
                        (-79.2915791686897506,      21.4033307581457670),
                        (-14.6577381710829471,      43.3895496964974043),
                        (-64.7472605264735108,     -21.8981713360336698),
                        (-29.0584693142401171,      43.2167287683090606),
                        (-72.0785319657452987,      -0.1815834632498404),
                        (-36.0366489745023770,      21.6135482886620949),
                        (-50.4808382862985496,      -7.3744722432402208),
                        (-50.5859026832315024,      21.5881966132975371),
                        (-0.1358203773809326,      28.7292896751977480),
                        (-65.0865638413727368,      36.0624693073746769),
                        (-21.4983260706612533,      -7.3194159498090388),
                        (-57.5687244704708050,      43.2505562436354225),
                        (-43.0700258454450875,     -14.5548396888330487)])

coords_list = np.array([(6734, 1453),
                        (2233,   10),
                        (5530, 1424),
                        (401, 841),
                        (3082, 1644),
                        (7608, 4458),
                        (7573, 3716),
                        (7265, 1268),
                        (6898, 1885),
                        (1112, 2049),
                        (5468, 2606),
                        (5989, 2873),
                        (4706, 2674),
                        (4612, 2035),
                        (6347, 2683),
                        (6107,  669),
                        (7611, 5184),
                        (7462, 3590),
                        (7732, 4723),
                        (5900, 3561),
                        (4483, 3369),
                        (6101, 1110),
                        (5199, 2182),
                        (1633, 2809),
                        (4307, 2322),
                        (675, 1006),
                        (7555, 4819),
                        (7541, 3981),
                        (3177,  756),
                        (7352, 4506),
                        (7545, 2801),
                        (3245, 3305),
                        (6426, 3173),
                        (4608, 1198),
                        (23, 2216),
                        (7248, 3779),
                        (7762, 4595),
                        (7392, 2244),
                        (3484, 2829),
                        (6271, 2135),
                        (4985,  140),
                        (1916, 1569),
                        (7280, 4899),
                        (7509, 3239),
                        (10, 2676),
                        (6807, 2993),
                        (5185, 3258),
                        (3023, 1942)])

edges_list = np.repeat(2, len(coords_list))

# Find "center of mass" of current list
com = np.mean(coords_list, axis=0)

seen = []

lines = []

while(sum(edges_list) > 1):
    # Figure out who's the furthest away from the c.o.m
    max_dist = 0
    max_point = (0, 0)
    max_index = -1

    for index, point in enumerate(coords_list):
        if(index in seen):
            continue

        dist = np.linalg.norm(point - com)
        if(dist > max_dist):
            max_dist = dist
            max_point = point
            max_index = index

    # Find who's the closest to that point
    min_dist = np.inf
    min_point = [0, 0]
    min_index = -1

    min_dist_2 = np.inf
    min_point_2 = [0, 0]
    min_index_2 = -1

    for index, point in enumerate(coords_list):
        if(index in seen):
            continue

        if (max_point == point).all():
            continue

        dist = np.linalg.norm(max_point - point)
        if(dist < min_dist):
            min_dist_2 = min_dist
            min_point_2 = min_point
            min_index_2 = min_index

            min_dist = dist
            min_point = point
            min_index = index
            continue

        if(dist < min_dist_2):
            min_dist_2 = dist
            min_point_2 = point
            min_index_2 = index
            continue

    if(edges_list[min_index] > 0):
        lines.append((max_point, min_point))

        edges_list[max_index] -= 1
        edges_list[min_index] -= 1

    if(edges_list[min_index_2] > 0 and edges_list[max_index] > 0):
        lines.append((max_point, min_point_2))

        edges_list[max_index] -= 1
        edges_list[min_index_2] -= 1

    seen.append(max_index)

    if(edges_list[min_index] == 0):
        seen.append(min_index)

    if(edges_list[min_index_2] == 0):
        seen.append(min_index_2)

total_dist = 0
for (p1, p2) in lines:
    total_dist += np.linalg.norm(p1 - p2)
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]])

plt.scatter(com[0], com[1])

print(total_dist)
plt.show()
