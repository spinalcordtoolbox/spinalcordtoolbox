

from multiprocessing import Pool
import multiprocessing
import numpy as np


def main():

    list = np.zeros(1000)
    list_result = f_mp(list)
    1


def f(list):
    for i in range(0, len(list)):
        list[i] = i**2
    return list

def f_mp(list):

    nb_pool = multiprocessing.cpu_count()

    chunks = [list[i::nb_pool] for i in range(nb_pool)]

    pool = Pool(processes=nb_pool)

    list = pool.map(f, chunks)

    return list



if __name__ == '__main__':

    main()
