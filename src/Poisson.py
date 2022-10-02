'''
Author: Yihan Liu
Date: 2022-10-01 21:03:29
LastEditTime: 2022-10-03 00:41:05
Email: 117010177@link.cuhk.edu.cn
'''
import numpy as np
import cv2
from scipy.sparse import lil_matrix as lil_matrix
from scipy.sparse import linalg


np.set_printoptions(threshold=np.inf)

def get_neighbor(index):
    i,j = index
    return [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
def create_lapla_matrix(N,omega):
    A = lil_matrix((N,N))
    for i,index in enumerate(omega):
        A[i,i] = 4
        for x in get_neighbor(index):
            if x not in omega: continue
            j = omega.index(x)
            A[i,j] = -1
    return A
def find_min(a):
    buf = 100000
    for i in a:
        if i < buf:
            buf = i
    return buf
def calculate_lapla(index,src,tar,off):
    i,j = index
    i2,j2 = i + off[0], j+off[1]
    lapla1 = (4 * src[i][j]) - src[i+1][j] - src[i-1][j] - src[i][j+1] - src[i][j-1]
    lapla2 = (4 * tar[i2][j2]) - tar[i2+1][j2] - tar[i2-1][j2] - tar[i2][j2+1] - tar[i2][j2-1]
    # if abs(lapla2) > abs(lapla1): return lapla2
    return (lapla1+lapla2)/2
def Poisson(src,mask,target,place_to_put):
    omega_src = np.nonzero(mask)
    top_left = [omega_src[0][0],find_min(omega_src[1])]
    offset = [place_to_put[0]-top_left[0],place_to_put[1]-top_left[1]]
    omega_src = zip(omega_src[0],omega_src[1])
    a = list(omega_src)
    pix_in_omega = len(a)
    # matrix for calculate the laplacian of f
    Lapla_f_A = create_lapla_matrix(pix_in_omega,a)
    result = []
    # handle the pic channel by channel
    for i in range(3):
        print(1)
        src_channel = src[:,:,i]
        target_channel = target[:,:,i]
        Lapla_g = np.zeros(pix_in_omega)
        for j,index in enumerate(a):
            Lapla_g[j] = calculate_lapla(index,src_channel,target_channel,offset)
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