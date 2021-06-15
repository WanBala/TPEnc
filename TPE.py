import matplotlib.pyplot as plt
import numpy as np
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def encrypt(img, iterations, block_size):
    print('Encrypting')
    height, width, channel = img.shape

    m, n = width // block_size, height // block_size

    data = img.reshape((-1,))

    totalRndForPermutation = iterations * n * m * block_size * block_size
    totalRndForSubstitution = iterations * n * m * (block_size * block_size - (block_size * block_size) % 2) // 2 * 3
    
    sAesRndNumGen = AesRndNumGen(totalRndForSubstitution)
    pAesRndNumGen = AesRndNumGen(totalRndForPermutation)
    
    for ccc in range(iterations):
        ### substitution
        for i in range(n):
            for j  in range(m):
                for k in range(0, block_size * block_size - 1, 2):
                    p = k // block_size
                    q = k % block_size
                    x = (k + 1) // block_size
                    y = (k + 1) % block_size

                    r1 = data[(i * width * block_size + p * width + j * block_size + q) * 3]
                    g1 = data[(i * width * block_size + p * width + j * block_size + q) * 3 + 1]
                    b1 = data[(i * width * block_size + p * width + j * block_size + q) * 3 + 2]


                    r2 = data[(i * width * block_size + x * width + j * block_size + y) * 3]
                    g2 = data[(i * width * block_size + x * width + j * block_size + y) * 3 + 1]
                    b2 = data[(i * width * block_size + x * width + j * block_size + y) * 3 + 2]

                    rt1 = sAesRndNumGen.getNewCouple(r1, r2, True)
                    rt2 = int(r1) + int(r2) - rt1

                    gt1 = sAesRndNumGen.getNewCouple(g1, g2, True)
                    gt2 = int(g1) + int(g2) - gt1

                    bt1 = sAesRndNumGen.getNewCouple(b1, b2, True)
                    bt2 = int(b1) + int(b2) - bt1

                    data[(i * width * block_size + p * width + j * block_size + q) * 3] = rt1
                    data[(i * width * block_size + p * width + j * block_size + q) * 3 + 1] = gt1
                    data[(i * width * block_size + p * width + j * block_size + q) * 3 + 2] = bt1

                    data[(i * width * block_size + x * width + j * block_size + y) * 3] = rt2
                    data[(i * width * block_size + x * width + j * block_size + y) * 3 + 1] = gt2
                    data[(i * width * block_size + x * width + j * block_size + y) * 3 + 2] = bt2

        ### Permutataion
        for i in range(n):
            for j in range(m):
                r_list = []
                g_list = []
                b_list = []
                for k in range(block_size * block_size):
                    p = k // block_size
                    q = k % block_size
                    r = data[(i * width * block_size + p * width + j * block_size + q) * 3]
                    g = data[(i * width * block_size + p * width + j * block_size + q) * 3 + 1]
                    b = data[(i * width * block_size + p * width + j * block_size + q) * 3 + 2]
                    r_list.append(r)
                    g_list.append(g)
                    b_list.append(b)

                permutation = pAesRndNumGen.getNewPermutation(block_size)
                    
                for k in range(block_size * block_size):
                    p = k // block_size
                    q = k % block_size
                    data[(i * width * block_size + p * width + j * block_size + q) * 3] = r_list[permutation[k]]
                    data[(i * width * block_size + p * width + j * block_size + q) * 3 + 1] = g_list[permutation[k]]
                    data[(i * width * block_size + p * width + j * block_size + q) * 3 + 2] = b_list[permutation[k]]
        
    data = data.reshape(img.shape)
    plt.figure()
    plt.title("Encrypted Image")
    plt.imshow(data)

    return data


def decrypt(img, num_of_iter, block_size):
    print('Decrypting')
    height, width, channel = img.shape

    m, n = width // block_size, height // block_size

    data = img.reshape((-1,))

    totalRndForPermutation = num_of_iter * n * m * block_size * block_size
    totalRndForSubstitution = num_of_iter * n * m * \
        (block_size * block_size - (block_size * block_size) % 2) // 2 * 3


    # gen key
    sAesRndNumGen = AesRndNumGen(totalRndForSubstitution)
    pAesRndNumGen = AesRndNumGen(totalRndForPermutation)

    pAesRndNumGen.ctr = totalRndForPermutation
    sAesRndNumGen.ctr = totalRndForSubstitution

    for ccc in range(num_of_iter):
        sAesRndNumGen.ctr = int((num_of_iter - (ccc + 1)) * (totalRndForSubstitution / num_of_iter))
        pAesRndNumGen.ctr = int((num_of_iter - (ccc + 1)) * (totalRndForPermutation / num_of_iter))

        # permutation reverse
        for i in range(n):
            for j in range(m):
                r_list = []
                g_list = []
                b_list = []
                for k in range(block_size * block_size):
                    p = k // block_size
                    q = k % block_size
                    r = data[(i * width * block_size + p * width + j * block_size + q) * 3]
                    g = data[(i * width * block_size + p * width + j * block_size + q) * 3 + 1]
                    b = data[(i * width * block_size + p * width + j * block_size + q) * 3 + 2]
                    r_list.append(r)
                    g_list.append(g)
                    b_list.append(b)

                permutation = pAesRndNumGen.getNewPermutation(block_size)

                for k in range(block_size * block_size):
                    p = permutation[k] // block_size
                    q = permutation[k] % block_size
                    data[(i * width * block_size + p * width + j * block_size + q) * 3] = r_list[k]
                    data[(i * width * block_size + p * width + j * block_size + q) * 3 + 1] = g_list[k]
                    data[(i * width * block_size + p * width + j * block_size + q) * 3 + 2] = b_list[k]

        # substitution reverse
        for i in range(n):
            for j in range(m):
                for k in range(0, block_size * block_size - 1, 2):
                    p = k // block_size
                    q = k % block_size
                    x = (k + 1) // block_size
                    y = (k + 1) % block_size

                    r1 = data[(i * width * block_size + p * width + j * block_size + q) * 3]
                    g1 = data[(i * width * block_size + p * width + j * block_size + q) * 3 + 1]
                    b1 = data[(i * width * block_size + p * width + j * block_size + q) * 3 + 2]

                    r2 = data[(i * width * block_size + x * width + j * block_size + y) * 3]
                    g2 = data[(i * width * block_size + x * width + j * block_size + y) * 3 + 1]
                    b2 = data[(i * width * block_size + x * width + j * block_size + y) * 3 + 2]

                    rt1 = sAesRndNumGen.getNewCouple(r1, r2, False)
                    rt2 = int(r1) + int(r2) - rt1

                    gt1 = sAesRndNumGen.getNewCouple(g1, g2, False)
                    gt2 = int(g1) + int(g2) - gt1

                    bt1 = sAesRndNumGen.getNewCouple(b1, b2, False)
                    bt2 = int(b1) + int(b2) - bt1

                    data[(i * width * block_size + p * width + j * block_size + q) * 3] = rt1
                    data[(i * width * block_size + p * width + j * block_size + q) * 3 + 1] = gt1
                    data[(i * width * block_size + p * width + j * block_size + q) * 3 + 2] = bt1

                    data[(i * width * block_size + x * width + j * block_size + y) * 3] = rt2
                    data[(i * width * block_size + x * width + j * block_size + y) * 3 + 1] = gt2
                    data[(i * width * block_size + x * width + j * block_size + y) * 3 + 2] = bt2
    data = data.reshape(img.shape)
    plt.figure()
    plt.title("Decrypted Image")
    plt.imshow(data)

    return data


class AesRndNumGen:
    def __init__(self, totalNeed):
        # print("AES init")
        self.ctr = 0
        self.data = np.zeros(totalNeed)

        # if not os.path.isfile('./key.txt'):
        #     key = generateKey()
        #     key_str = exportKey(key)
        #     with open('./key.txt') as f:
        #         f.write(key_str)
        # with open('./key.txt') as f:
        #     key_str = f.readline()
        # print('key:', key_str)
        # key = importKey(key_str)
        # encrypt(key, data)

    def next(self):
        self.ctr += 1
        # print(ctr)
        return self.data[self.ctr - 1]

    def getNewCouple(self, p, q, enc):
        rnd = self.next()
        sum = int(p) + int(q)
        if sum <= 255:
            if enc:
                rnd = (p + rnd) % (sum + 1)
            else:
                rnd = (p - rnd) % (sum + 1)
            if rnd < 0:
                rnd = rnd + sum + 1
            return rnd
        else:
            if enc:
                rnd = 255 - (p + rnd) % (511 - sum)
                return rnd
            else:
                rnd = (255 - p - rnd) % (511 - sum)
                while rnd < (sum - 255):
                    rnd += 511 - sum
                return rnd

    def getNewPermutation(self, block_size):
        permutation = []
        for _ in range(block_size * block_size):
            permutation.append(self.next())
        len = block_size * block_size
        indices = [i for i in range(len)]
        indices.sort()
        return indices


if __name__ == '__main__':
    iterations = 1
    blocksize = 200
    print(f'block_size: {blocksize}, iterations: {iterations}')

    Tk().withdraw()
    filename = askopenfilename()

    img = plt.imread(filename, 0)
    plt.figure()
    plt.title("Original Image")
    plt.imshow(img)
    img = img.copy()

    ans = input('[E]ncrypt or [D]ecrypt? ')
    if ans == 'E':
        enImg = encrypt(img, iterations, blocksize)
        plt.imsave('encrypt.png', enImg)
    elif ans == 'D':
        deImg = decrypt(img, iterations, blocksize)
        plt.imsave('decrypt.png', deImg)
    else:
        print('No such operation')
    plt.show()