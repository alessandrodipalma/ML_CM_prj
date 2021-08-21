import numpy as np


def swap_rows(matrix, r1, r2):
    matrix[[r1, r2], :] = matrix[[r2, r1], :]  # swap row 0 with row 4...
    # matrix[:, [0, 4]] = matrix[:, [4, 0]]  # ...and column 0 with column 4
    return matrix


def bring_to_top(matrix, start, end):
    orig = list(range(0, end))
    moved = list(range(start, end)) + list(range(0, start))
    matrix[orig, :] = matrix[moved, :]
    matrix[:, orig] = matrix[:, moved]
    return matrix

def swap_indexes(matrix, from_where, to_where):
    matrix[[from_where, to_where], :] = matrix[[to_where, from_where], :]
    matrix[:, [from_where, to_where]] = matrix[:, [to_where, from_where]]
    return matrix

def swap_row_col(matrix, from_w, to_w):
    matrix = swap_indexes(matrix, from_w, to_w)

def split_kernel(matrix, qbb_start, qbb_end):
    matrix = bring_to_top(matrix, qbb_start, qbb_end)
    qbb = matrix[0:qbb_end - qbb_start, 0:qbb_end - qbb_start]
    qnb = matrix[qbb_end - qbb_start:, 0:qbb_end - qbb_start]
    qnn = matrix[qbb_end - qbb_start:, qbb_end - qbb_start:]

    return qbb, qnb, qnn

def split_kernel_working(matrix, working_indexes):
    matrix = move_forward(matrix, working_indexes)
    matrix = split_kernel(matrix, 0, len(working_indexes))
    return matrix
def move_forward(matrix, indexes):
    for from_w, to_w in enumerate(indexes):
        matrix = swap_indexes(matrix, from_w, to_w)
    return matrix

def get_working_part (array, working_indexes):
    xb = array[working_indexes]
    xn = np.delete(array, working_indexes, axis=0)
    return xb, xn


def split_alpha(x, qbb_start, qbb_end):
    n = int(x.shape[0] / 2)
    half = n
    xb = np.append(x[qbb_start:qbb_end], x[half + qbb_start: half + qbb_end])
    xn = np.concatenate([x[0:qbb_start],
                         x[qbb_end:half],
                         x[half: half + qbb_start],
                         x[half + qbb_end:]])

    return xb, xn


def update_alpha(x, xb, qbb_start, qbb_end):
    n_half = int(x.shape[0] / 2)
    half = int(xb.shape[0] / 2)

    x[qbb_start:qbb_end] = xb[:half]
    x[n_half + qbb_start: n_half + qbb_end] = xb[half:]
    return x


# matrix = np.array([[11, 12, 13, 14],
#                    [21, 22, 23, 24],
#                    [31, 32, 33, 34],
#                    [41, 42, 43, 44]])
#
# print(move_forward(matrix, [1,3]))
# Qbb, Qnb, Qnn = split_kernel(matrix, 1,3)
# print(Qbb)
# print(Qnb)
# print(Qnn)

x = np.array(list(range(10)) + list(range(10)))

xb, xn = get_working_part(x, [5, 8])
print(xb)
print(xn)

# nsp = 3
# x = np.array(list(range(10)) + list(range(10)))
# xb = np.zeros(nsp*2)
# print(x, xb)
# print(update_alpha(x, xb, 5,8))

# a = np.array([1, 2, 3, 4])
# a1 = np.array([1,2])
# a2 = np.array([3,4])
#
# Q1 = np.array([[11,12],
#                [12,22]])
#
# Q = np.block([[Q1,-Q1],[-Q1, Q1]])
# print(Q)
# print(a.T@Q@a)
# print(a1.T@Q1@a1.T - 2 * a1.T@Q1@a2 + a2.T@Q1@a2)
#
# a11 = np.array([1,3])
# a22 = np.array([2,4])
# QB = np.array([[11,-11],
#                [-11,11]])
# Qbn = np.array([[12,-12],[-12,12]])
# Qnn = np.array([[22,-22],[-22,22]])
#
# print(a11.T @ QB @ a11 + 2*a22.T@Qbn@a11 + a22.T@Qnn@a22)
