'''
Author: Yihan Liu
Date: 2022-10-01 21:03:29
LastEditTime: 2022-10-04 17:58:44
Email: 117010177@link.cuhk.edu.cn
'''
import numpy as np
import cv2
from scipy.sparse import lil_matrix as lil_matrix
from scipy.sparse import linalg


np.set_printoptions(threshold=np.inf)
'''
description: 
get the neibourhood of index 
param {*} index
return {*}
'''
def get_neighbor(index):
    i,j = index
    return [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
'''
description: 
create the lapla f matrix for linear problem
param {*} N
param {*} omega
param {*} mask
return {*}
'''
def create_lapla_matrix(N,omega,mask):
    A = lil_matrix((N,N))
    for i,index in enumerate(omega):
        # print(i)
        A[i,i] = 4
        for x in get_neighbor(index):
            if mask[x] == 0: continue
            j = omega.index(x)
            A[i,j] = -1
    return A
'''
description:
find the minimum value in a  
param {*} a
return {*}
'''
def find_min(a):
    buf = 1000000
    for i in a:
        if i < buf:
            buf = i
    return buf
'''
description: 
calculate the laplacian at index, vpq is depends on the gray image
param {*} index
param {*} src
param {*} tar
param {*} g_src
param {*} g_tar
param {*} off
return {*}
'''
def calculate_lapla(index,src,tar,g_src,g_tar,off):
    i,j = index
    i2,j2 = i + off[0], j+off[1]
    # lapla1 = (4 * src[i][j]) - src[i+1][j] - src[i-1][j] - src[i][j+1] - src[i][j-1]
    # lapla2 = (4 * tar[i2][j2]) - tar[i2+1][j2] - tar[i2-1][j2] - tar[i2][j2+1] - tar[i2][j2-1]
    # lapla1_g = (4 * g_src[i][j]) - g_src[i+1][j] - g_src[i-1][j] - g_src[i][j+1] - g_src[i][j-1]
    # lapla2_g = (4 * g_tar[i2][j2]) - g_tar[i2+1][j2] - g_tar[i2-1][j2] - g_tar[i2][j2+1] - g_tar[i2][j2-1]
    # if abs(lapla2_g) > abs(lapla1_g): return lapla2
    lapla = 0
    if abs(g_src[i][j] - g_src[i+1][j]) < abs(g_tar[i2][j2] - g_tar[i2+1][j2]):
        lapla += tar[i2][j2] - tar[i2+1][j2]
    else:
        lapla += src[i][j] - src[i+1][j]
        
    if abs(g_src[i][j] - g_src[i-1][j]) < abs(g_tar[i2][j2] - g_tar[i2-1][j2]):
        lapla += tar[i2][j2] - tar[i2-1][j2]
    else:
        lapla += src[i][j] - src[i-1][j]
        
        
    if abs(g_src[i][j] - g_src[i][j+1]) < abs(g_tar[i2][j2] - g_tar[i2][j2+1]):
        lapla += tar[i2][j2] - tar[i2][j2+1]
    else:
        lapla += src[i][j] - src[i][j+1]
        
        
    if abs(g_src[i][j] - g_src[i][j-1]) < abs(g_tar[i2][j2] - g_tar[i2][j2-1]):
        lapla += tar[i2][j2] - tar[i2][j2-1]
    else:
        lapla += src[i][j] - src[i][j-1]
    return lapla





'''
description: 
Generate the seamless cloning by solving poisson equation

param {*} src
param {*} mask
param {*} target
param {*} gray_src
param {*} gray_tar
param {*} place_to_put
return {*}

'''
def Poisson(src,mask,target,gray_src,gray_tar,place_to_put):
    omega_src = np.nonzero(mask)
    top_left = [omega_src[0][0],find_min(omega_src[1])]
    offset = [place_to_put[0]-top_left[0],place_to_put[1]-top_left[1]]
    omega_src = zip(omega_src[0],omega_src[1])
    a = list(omega_src)
    pix_in_omega = len(a)
    # matrix for calculate the laplacian of f
    Lapla_f_A = create_lapla_matrix(pix_in_omega,a,mask)
    result = []
    # handle the pic channel by channel
    for i in range(3):
        src_channel = src[:,:,i]
        target_channel = target[:,:,i]
        Lapla_g = np.zeros(pix_in_omega)
        for j,index in enumerate(a):
            Lapla_g[j] = calculate_lapla(index,src_channel,target_channel,gray_src,gray_tar,offset)
            for n in get_neighbor(index):
                if mask[n] == 0:
                    x,y = n
                    Lapla_g[j] += target_channel[x+offset[0]][y+offset[1]]
        m = linalg.cg(Lapla_f_A, Lapla_g)
        result.append(np.copy(target_channel))
        for j,index in enumerate(a):
            x,y = index
            result[i][x+offset[0]][y+offset[1]] = m[0][j]
    result = cv2.merge(result)
    return result