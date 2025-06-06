# MIT License

# Thanks for 2022 Siyou Lin, Dong Xiao, Zuoqiang Shi, Bin Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree
from time import time
import math


def load_sample_from_npy(file_path, return_cupy, dtype):
    data = np.load(file_path)
    data = data.astype(dtype)
    if return_cupy:
        import cupy as cp
        data = cp.array(data)
    return data

def mul_A_T(x, y, xi, x_width, chunk_size, dtype,c):
    if isinstance(x, np.ndarray):
        cnp = np
    else:
        import cupy as cnp
 
    N_query = x.shape[0]
    N_sample = y.shape[0]
    n_y_chunks = N_sample // chunk_size + 1

    c1=c[:, 0].reshape(-1, 1)
    c2=c[:, 1].reshape(-1, 1)
    c3=c[:, 2].reshape(-1, 1)
    
    lse = cnp.zeros((3, N_sample), dtype=dtype)
    for i in range(n_y_chunks):
        y_chunk = y[i*chunk_size:(i+1)*chunk_size]
        A1_chunk = get_A2(x, y_chunk, x_width,c1)
        A2_chunk = get_A2(x, y_chunk, x_width,c2)
        A3_chunk = get_A2(x, y_chunk, x_width,c3)
        A_chunk =cnp.concatenate((A1_chunk, A2_chunk, A3_chunk), axis=0)
        lse[0, i*chunk_size:(i+1)*chunk_size], lse[1, i*chunk_size:(i+1)*chunk_size], lse[2, i*chunk_size:(i+1)*chunk_size] = cnp.einsum('jk,j', A_chunk, xi).reshape(3, -1)


    return lse.reshape(-1)

def get_A(x, y, x_width):
    """
    x: numpy/cupy array of shape [N_query, 3]
    y: numpy/cupy array of shape [N_sample, 3]
    x_width: [N_query]
    ---
    return:
    A: numpy array of shape [N_query, 3 * N_sample]
    """

    if isinstance(x, np.ndarray):
        cnp = np
    else:
        import cupy as cnp
    
    N_query = x.shape[0]
    
    A = x[:, None] - y[None, :]            


    dist = cnp.sqrt((A ** 2).sum(-1))       
    inv_dist = cnp.where(dist > x_width[:, None], 1/dist, 1 / x_width[:, None]) 

    inv_cub_dist = inv_dist ** 3 / 4 / cnp.pi 

 

    A *= inv_cub_dist[..., None]    
    A = A.transpose((0,2,1))
    A = A.reshape(N_query, -1)
    return -A


def get_A3(x, y, x_width,c):
   

    if isinstance(x, np.ndarray):
        cnp = np
    else:
        import cupy as cnp
    
    N_query = x.shape[0]
    A0 = x[:, None] - y[None, :]             
    cnorm = np.linalg.norm(c)
    cxy = cnp.matmul(A0, c)
    dist = cnp.sqrt((A0 ** 2).sum(-1))       
    inv_dist = cnp.where(dist > x_width[:, None], 1/dist, 1 / x_width[:, None]) # / 4 / cp.pi
    smooth_dist=cnp.where(dist > x_width[:, None], dist, x_width[:, None]) 
    cdist=cnorm *smooth_dist
    cdist3=cnp.expand_dims(cdist, axis=2)
    c1=1/(4*cnp.pi)*cnp.exp(1/2*(cxy-cdist3))
    c2=inv_dist
    inv_cub_dist = inv_dist ** 3 / 4 / cnp.pi 
    A1=inv_dist[..., None]** 2*A0
    expanded_matrix = np.repeat(c, x.shape[0], axis=0)
    expanded_matrix = np.repeat(expanded_matrix, y.shape[0], axis=1)
    c_expand = expanded_matrix.reshape(x.shape[0], y.shape[0], 3)
    A2=0.5*c_expand
    A3=0.5*inv_dist[..., None] *A0*cnorm
    A=(A1+A2+A3)*c1*c2[..., None]
    A = A.transpose((0,2,1))
    A = A.reshape(N_query, -1)

    return -A


def get_A2(x, y, x_width,c):


    if isinstance(x, np.ndarray):
        cnp = np
    else:
        import cupy as cnp
    
    N_query = x.shape[0]

    
    
    A0 = x[:, None] - y[None, :]             # [N_query, N_sample, 3]
    cnorm = np.linalg.norm(c)
    cxy = cnp.matmul(A0, c)
    abc=c[0]*c[1]*c[2]
    bcacab=cnp.array([c[1]*c[2],c[0]*c[2],c[0]*c[1]])
    dist = cnp.sqrt(cnp.matmul(A0 ** 2,bcacab).sum(-1))        # [N_query, N_sample], |x^i-y^j|^2
    inv_dist = cnp.where(dist > x_width[:, None], 1/dist, 1 / x_width[:, None]) # / 4 / cp.pi
    smooth_dist=cnp.where(dist > x_width[:, None], dist, x_width[:, None]) 
    cdist=cnorm *smooth_dist
    cdist3=cnp.expand_dims(cdist, axis=2)


    c1=1/(4*cnp.pi)*c[0]*c[1]*c[2]
    c2=inv_dist

    A1=inv_dist[..., None]** 3*A0

    A=A1*c1[..., None]
    
    A = A.transpose((0,2,1))
    A = A.reshape(N_query, -1)

    return -A




def get_B(x, y, chunk_size, x_width, alpha,c):
    """
    x: numpy array of shape [N_query, 3]
    y: numpy array of shape [N_sample, 3]
    ---
    return:
    B: numpy array of shape [N_query, N_query], which is AA^T
    """
    if isinstance(x, np.ndarray):
        cnp = np
    else:
        import cupy as cnp

    
    N_query = x.shape[0]
    N_sample = y.shape[0]

    num_columns = c.shape[0]
    
    B = cnp.zeros((N_query*num_columns, N_query*num_columns), dtype=x.dtype)

    c1=c[:, 0].reshape(-1, 1)
    c2=c[:, 1].reshape(-1, 1)
    c3=c[:, 2].reshape(-1, 1)
 
    
    
    n_row_chunks = N_query // chunk_size + 1
    n_col_chunks = n_row_chunks
    
    for i in tqdm(range(n_row_chunks)):
        x_chunk_i = x[i*chunk_size:(i+1)*chunk_size]
        
        if x_chunk_i.shape[0] <= 0:
            continue



        x_chunk_eps_i = x_width[i*chunk_size:(i+1)*chunk_size]
        A_block1_i = get_A2(x_chunk_i, y, x_chunk_eps_i,c1)
        A_block2_i = get_A2(x_chunk_i, y, x_chunk_eps_i,c2)
        A_block3_i = get_A2(x_chunk_i, y, x_chunk_eps_i,c3)
     
        

        for j in range(0, n_col_chunks):
            x_chunk_j = x[j*chunk_size:(j+1)*chunk_size]
            
            if x_chunk_j.shape[0] <= 0:
                continue
            
            x_chunk_eps_j = x_width[j*chunk_size:(j+1)*chunk_size]
            A_block1_j = get_A2(x_chunk_j, y, x_chunk_eps_j,c1)
            A_block2_j = get_A2(x_chunk_j, y, x_chunk_eps_j,c2)
            A_block3_j = get_A2(x_chunk_j, y, x_chunk_eps_j,c3)


            if j>=i :
                B[i*chunk_size:(i+1)*chunk_size, j*chunk_size:(j+1)*chunk_size] = cnp.einsum('ik,jk->ij', A_block1_i, A_block1_j)
                B[i*chunk_size:(i+1)*chunk_size, j*chunk_size + N_query:(j+1)*chunk_size+ N_query] = cnp.einsum('ik,jk->ij', A_block1_i, A_block2_j)
                B[i*chunk_size:(i+1)*chunk_size, j*chunk_size + 2*N_query:(j+1)*chunk_size+ 2*N_query] = cnp.einsum('ik,jk->ij', A_block1_i, A_block3_j)
                B[i*chunk_size + N_query:(i+1)*chunk_size + N_query, j*chunk_size + N_query:(j+1)*chunk_size + N_query] = cnp.einsum('ik,jk->ij', A_block2_i, A_block2_j)
                B[i*chunk_size + N_query:(i+1)*chunk_size + N_query, j*chunk_size + 2*N_query:(j+1)*chunk_size + 2*N_query] = cnp.einsum('ik,jk->ij', A_block2_i, A_block3_j)
                B[i*chunk_size + 2*N_query:(i+1)*chunk_size + 2*N_query, j*chunk_size + 2*N_query:(j+1)*chunk_size + 2*N_query] = cnp.einsum('ik,jk->ij', A_block3_i, A_block3_j)
    
            else:
                B[i*chunk_size:(i+1)*chunk_size, j*chunk_size + N_query:(j+1)*chunk_size+ N_query] = cnp.einsum('ik,jk->ij', A_block1_i, A_block2_j)
                B[i*chunk_size:(i+1)*chunk_size, j*chunk_size + 2*N_query:(j+1)*chunk_size+ 2*N_query] = cnp.einsum('ik,jk->ij', A_block1_i, A_block3_j)
                B[i*chunk_size + N_query:(i+1)*chunk_size + N_query, j*chunk_size + 2*N_query:(j+1)*chunk_size + 2*N_query] = cnp.einsum('ik,jk->ij', A_block2_i, A_block3_j)

           
            
            if j>i:
                B[j*chunk_size:(j+1)*chunk_size, i*chunk_size:(i+1)*chunk_size] = B[i*chunk_size:(i+1)*chunk_size, j*chunk_size:(j+1)*chunk_size].T
                B[j*chunk_size + N_query:(j+1)*chunk_size + N_query, i*chunk_size:(i+1)*chunk_size] = B[i*chunk_size:(i+1)*chunk_size, j*chunk_size + N_query:(j+1)*chunk_size + N_query].T
                B[j*chunk_size + N_query:(j+1)*chunk_size + N_query, i*chunk_size + N_query:(i+1)*chunk_size + N_query] = B[i*chunk_size + N_query:(i+1)*chunk_size + N_query, j*chunk_size + N_query:(j+1)*chunk_size + N_query].T
                B[j*chunk_size + 2*N_query:(j+1)*chunk_size + 2*N_query, i*chunk_size:(i+1)*chunk_size] = B[i*chunk_size:(i+1)*chunk_size, j*chunk_size + 2*N_query:(j+1)*chunk_size + 2*N_query].T
                B[j*chunk_size + 2*N_query:(j+1)*chunk_size + 2*N_query, i*chunk_size + N_query:(i+1)*chunk_size+ N_query] = B[i*chunk_size + N_query:(i+1)*chunk_size + N_query, j*chunk_size + 2*N_query:(j+1)*chunk_size + 2*N_query].T
                B[j*chunk_size + 2*N_query:(j+1)*chunk_size + 2*N_query, i*chunk_size + 2*N_query:(i+1)*chunk_size + 2*N_query] = B[i*chunk_size + 2*N_query:(i+1)*chunk_size + 2*N_query, j*chunk_size + 2*N_query :(j+1)*chunk_size + 2*N_query].T
            else:
                
                B[j*chunk_size + N_query:(j+1)*chunk_size + N_query, i*chunk_size:(i+1)*chunk_size] = B[i*chunk_size:(i+1)*chunk_size, j*chunk_size + N_query:(j+1)*chunk_size + N_query].T
                B[j*chunk_size + 2*N_query:(j+1)*chunk_size + 2*N_query, i*chunk_size:(i+1)*chunk_size] = B[i*chunk_size:(i+1)*chunk_size, j*chunk_size + 2*N_query:(j+1)*chunk_size + 2*N_query].T
                B[j*chunk_size + 2*N_query:(j+1)*chunk_size + 2*N_query, i*chunk_size + N_query:(i+1)*chunk_size+ N_query] = B[i*chunk_size + N_query:(i+1)*chunk_size + N_query, j*chunk_size + 2*N_query:(j+1)*chunk_size + 2*N_query].T
               
            if i == j:
                block_size = B[i*chunk_size:(i+1)*chunk_size, j*chunk_size:(j+1)*chunk_size].shape[0]
                if block_size <= 0:
                    continue
                diag_mask = cnp.eye(block_size, dtype=bool)
           
                B[i*chunk_size:(i+1)*chunk_size, j*chunk_size:(j+1)*chunk_size][diag_mask] *= alpha
                B[i*chunk_size + N_query:(i+1)*chunk_size+ N_query, j*chunk_size + N_query:(j+1)*chunk_size+ N_query][diag_mask] *= alpha
                B[i*chunk_size + 2*N_query:(i+1)*chunk_size+ 2*N_query, j*chunk_size + 2*N_query:(j+1)*chunk_size+ 2*N_query][diag_mask] *= alpha

               
    return B

def get_B2(x, y, chunk_size, x_width, alpha,c):


    if isinstance(x, np.ndarray):
        cnp = np
    else:
        import cupy as cnp

    
    N_query = x.shape[0]
    N_sample = y.shape[0]
    N_query=N_sample

    num_columns = c.shape[1]

    
    B = cnp.zeros((N_query*3, N_query*3), dtype=x.dtype)
    print(B.shape)

    
    
    n_row_chunks = N_query // chunk_size
    n_col_chunks = n_row_chunks
    for j in range(num_columns):
        for i in tqdm(range(n_row_chunks)):
            x_chunk_i = x[i*chunk_size:(i+1)*chunk_size]
        
            if x_chunk_i.shape[0] <= 0:
                continue
            x_chunk_eps_i = x_width[i*chunk_size:(i+1)*chunk_size]
            A_block_j = get_A2(x_chunk_i, y, x_chunk_eps_i,c[:,j].reshape(-1, 1))
            B=B+cnp.einsum('ki,kj->ij', A_block_j, A_block_j)

    block_size = B.shape[0]
    diag_mask = cnp.eye(block_size, dtype=bool)
    B[diag_mask] *= alpha


    return B

def solve2(x, y, x_width, chunk_size, dtype, iso_value, r_sq_stop_eps, alpha, max_iters, save_r,c):
    """
    x: numpy array of shape [N_query, 3]
    y: numpy array of shape [N_sample, 3]
    ---
    return:
    lse:
    """

    if isinstance(x, np.ndarray):
        cnp = np
    else:
        import cupy as cnp  

    num_columns = c.shape[1]

    
    N_query = x.shape[0]
    N_sample = y.shape[0]
    N_query=N_sample

    if max_iters is None:
        max_iters = y.shape[0]
    else:
        max_iters = min(max_iters, y.shape[0])


    print(f'[In solver] Precomputing B...')
    TIME_START_COMPUTE_B = time()
    B = get_B2(x, y, chunk_size, x_width, alpha,c)
    TIME_END_COMPUTE_B = time()
    print('\033[94m' + f'[Timer] B computed in {TIME_END_COMPUTE_B-TIME_START_COMPUTE_B}' + '\033[0m')
    
    if cnp != np:
        cnp._default_memory_pool.free_all_blocks()
    
    xi = cnp.zeros(N_query*3, dtype=dtype)
    r0 = cnp.ones(N_query*num_columns, dtype=dtype) * iso_value
    n_row_chunks = N_query // chunk_size
    n_col_chunks = n_row_chunks

    for j in range(num_columns):
        for i in tqdm(range(n_row_chunks)):
            x_chunk_i = x[i*chunk_size:(i+1)*chunk_size]
        
            if x_chunk_i.shape[0] <= 0:
                continue

            x_chunk_eps_i = x_width[i*chunk_size:(i+1)*chunk_size]
            A0 = get_A2(x_chunk_i, y, x_chunk_eps_i,c[:,j].reshape(-1, 1)).T
            if (i+j) ==0:
                AT=A0
            else:
                AT= cnp.concatenate((AT, A0), axis=1)
    print(AT.shape)
    print(r0.shape)
    r = cnp.matmul(AT, r0)

    p = r.copy()

    if save_r:
        r_list = []
    
    print('[In solver] Starting CG iterations...')
    TIME_START_CG = time()
    solve_progress_bar = tqdm(range(max_iters))
    k = -1 
    for k in solve_progress_bar:
        Bp = cnp.matmul(B, p)
        r_sq = cnp.einsum('i,i', r, r)
        
        alpha = r_sq / cnp.einsum('i,i', p, Bp)
        xi += (alpha * p)
        r -= (alpha * Bp)
        beta = cnp.einsum('i,i', r, r) / r_sq
        p *= beta
        p += r
        
        solve_progress_bar.set_description(f'[In solver] {r_sq.item():.2e}/{r_sq_stop_eps:.0e}')
        

        if save_r:
            r_list.append(math.sqrt(r_sq.item()))

        if r_sq < r_sq_stop_eps:
            solve_progress_bar.close()
            break
    print('', end='')
    print(f'[In solver] Converged in {k+1}/{max_iters} iterations')  


    lse=xi
    print(f'[In solver] Got linearized surface elements')
    TIME_END_CG = time()

    print('\033[94m' + f'[Timer] CG finished in {TIME_END_CG-TIME_START_CG}' + '\033[0m')
    
    if save_r:
        return lse, r_list
    return lse


def solve(x, y, x_width, chunk_size, dtype, iso_value, r_sq_stop_eps, alpha, max_iters, save_r,c):
    """
    x: numpy array of shape [N_query, 3]
    y: numpy array of shape [N_sample, 3]
    ---
    return:
    lse:
    """

    if isinstance(x, np.ndarray):
        cnp = np
    else:
        import cupy as cnp  

    num_columns = c.shape[0]

    
    N_query = x.shape[0]
    N_sample = y.shape[1]

    if max_iters is None:
        max_iters = y.shape[0]
    else:
        max_iters = min(max_iters, y.shape[0])


    print(f'[In solver] Precomputing B...')
    TIME_START_COMPUTE_B = time()
    B = get_B(x, y, chunk_size, x_width, alpha,c)
    TIME_END_COMPUTE_B = time()
    print('\033[94m' + f'[Timer] B computed in {TIME_END_COMPUTE_B-TIME_START_COMPUTE_B}' + '\033[0m')
    
    if cnp != np:
        cnp._default_memory_pool.free_all_blocks()
    
    xi = cnp.zeros(N_query*num_columns, dtype=dtype)
    r = cnp.ones(N_query*num_columns, dtype=dtype) * iso_value
    p = r.copy()

    if save_r:
        r_list = []
    
    print('[In solver] Starting CG iterations...')
    TIME_START_CG = time()
    solve_progress_bar = tqdm(range(max_iters))
    k = -1 
    for k in solve_progress_bar:
        Bp = cnp.matmul(B, p)
        r_sq = cnp.einsum('i,i', r, r)
        
        alpha = r_sq / cnp.einsum('i,i', p, Bp)
        xi += (alpha * p)
        r -= (alpha * Bp)
        beta = cnp.einsum('i,i', r, r) / r_sq
        p *= beta
        p += r
        
        solve_progress_bar.set_description(f'[In solver] {r_sq.item():.2e}/{r_sq_stop_eps:.0e}')
        

        if save_r:
            r_list.append(math.sqrt(r_sq.item()))

        if r_sq < r_sq_stop_eps:
            solve_progress_bar.close()
            break
    print('', end='')
    print(f'[In solver] Converged in {k+1}/{max_iters} iterations')  
 
    lse = mul_A_T(x, y, xi, x_width, chunk_size, dtype,c)
    print(f'[In solver] Got linearized surface elements')
    TIME_END_CG = time()

    print('\033[94m' + f'[Timer] CG finished in {TIME_END_CG-TIME_START_CG}' + '\033[0m')
    
    if save_r:
        return lse, r_list
    return lse

def get_query_vals(queries, q_width, y_base, lse, chunk_size,c):
    if isinstance(y_base, np.ndarray):
        cnp = np
    else:
        import cupy as cnp
    
    query_chunks = np.array_split(queries, len(queries)//chunk_size + 1)
    q_cut_chunks = np.array_split(q_width, len(queries)//chunk_size + 1)
    query_vals = []
    c1=c[:, 0].reshape(-1, 1)
    c2=c[:, 1].reshape(-1, 1)
    c3=c[:, 2].reshape(-1, 1)
    print(f'[In solver] Starting to query on the grid...')
    tqdmbar_query = tqdm(list(zip(query_chunks, q_cut_chunks)))

    for chunk, cut_chunk in tqdmbar_query:
        chunk = cnp.array(chunk)
        cut_chunk = cnp.array(cut_chunk)
        A_show1 = get_A2(chunk, y_base, cut_chunk,c1)
        A_show2 = get_A2(chunk, y_base, cut_chunk,c2)
        A_show3 = get_A2(chunk, y_base, cut_chunk,c3)
        A_show = (A_show1+A_show2+A_show3)/3
        query_vals.append( cnp.matmul(A_show, lse).get() if cnp != np else cnp.matmul(A_show, lse) )
    
    query_vals = np.concatenate(query_vals, axis=0)
    query_vals= query_vals.astype(np.float32)
    return query_vals

def get_width(query_set, k, dtype, width_min, width_max, base_set=None, base_kdtree=None, return_kdtree=False):
    assert not (base_set is None and base_kdtree is None)

    if base_kdtree is None:
        base_kdtree = KDTree(base_set)

    x_knn_dist, x_knn_idx = base_kdtree.query(query_set, k=k+1) # [N, k],
    x_knn_dist = x_knn_dist.astype(dtype)
    
    x_width = np.sqrt(np.einsum('nk,nk->n', x_knn_dist[:, 1:], x_knn_dist[:, 1:]) / k)
    x_width[x_width > width_max] = width_max
    x_width[x_width < width_min] = width_min


    if return_kdtree:
        return x_width, base_kdtree
    else:
        del base_kdtree
        return x_width

