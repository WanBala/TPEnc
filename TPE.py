import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import secrets
import itertools
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from exif import Image
# import Crypto.Cipher.AES as AES

def plot(img, title_name):
    plt.figure()
    plt.title(title_name)
    plt.imshow(img)


def thumbnail(img, block_size, img_name):
    height, width, channel = img.shape
    m, n = width // block_size, height // block_size
    data = img.reshape((-1,)).copy()
    for i in range(n):
        for j in range(m):
            r = 0
            g = 0
            b = 0
            for k in range(block_size * block_size):
                p = k // block_size
                q = k % block_size
                r += data[(i * width * block_size + p * width + j * block_size + q) * 3]
                g += data[(i * width * block_size + p * width + j * block_size + q) * 3 + 1]
                b += data[(i * width * block_size + p * width + j * block_size + q) * 3 + 2]

            for k in range(block_size * block_size):
                p = k // block_size
                q = k % block_size
                data[(i * width * block_size + p * width + j * block_size + q) * 3] = r // (block_size * block_size)
                data[(i * width * block_size + p * width + j * block_size + q) * 3 + 1] = g // (block_size * block_size)
                data[(i * width * block_size + p * width + j * block_size + q) * 3 + 2] = b // (block_size * block_size)
    data = data.reshape(img.shape)
    plot(data, img_name)
    
    return data


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
        # substitution
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

        # permutataion
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
    plot(data, "Encrypted Image")

    md = {'model' : 'TPE/{0}/{1}'.format(iterations, block_size)}
    filename = 'encrypt.png'
    #plt.imsave('encrypt.png', data, metadata=md)
    im = PIL.Image.fromarray(data)
    im.save("encrypt.png")
    #plt.imsave('encrypt.png', data)

    editor = exif_editor(filename)
    editor.set_etif(iterations, block_size)
    editor.save()
    return data


def decrypt(img, num_of_iter, block_size):
    print('Decrypting')

    #editor = exif_editor('encrypt.png')
    #if editor.is_valid == False:
    #    raise Exception("Tag message isn't included")
    img_data = PIL.Image.open('encrypt.png')
    data = img_data._getexif()[272]
    
    block_size, num_of_iter = exif_editor.check_valid(data)
    print("block_size: {}, iterations: {}".format(block_size, num_of_iter))
    print(block_size, num_of_iter)
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
    plot(data, "Decrypted Image")

    plt.imsave('decrypt.png', data)
    
    return data


class AesRndNumGen:
    def __init__(self, totalNeed):
        # print("AES init")
        print(totalNeed)
        self.ctr = 0
        self.data = np.zeros(totalNeed)
        self.data_length = totalNeed

        if not os.path.isfile('data.npy'):
            data = np.random.randint(1e3, size=totalNeed)
            # key_str = self.generateKey()
            # key_str = self.exportKey(key)
            # with open('./key.txt', "w", encoding='utf-8') as f:
            #     f.write(str(self.data))
            np.save('data.npy', data)

        # with open('./key.txt', "r", encoding='utf-8') as f:
        #     self.data = int(f.readline())
        self.data = np.load('data.npy')

        # print('key:', key_str)
        # # self.importKey(key_str)
        # encrypt(key)


    def generateKey(self):
        return secrets.token_hex(32)

    def encrypt(self, key):
        aes = AES.new(key, AES.MODE_CTR)
        ct_byte = aes.encrypt(self.data)
        print(type(ct_byte))
        print(len(ct_byte))
        print(ct_byte)

    def importKey(self, key_str):
        key_cycles = itertools.cycle(key_str)
        for i in range(self.data_length):
            self.data[i] = int(next(key_cycles), 16)


    def exportKey(self, key):
        pass

    def next(self):
        self.ctr += 1
        # print(self.ctr)
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
        permutation, indices = zip(*sorted(zip(permutation, indices)))
        return indices


class exif_editor:
    def __init__(self, filename):
        self._filename = filename
        with open(filename, "rb") as stream:
            self._exif = Image(stream)

        self.has_exif = self._exif.has_exif
        self.key_message = self._exif['model'] if self.has_exif else None
        self.block_size = None
        self.iterations = None

        if self.has_exif != False:
            try:
                split = self.key_message.split("/")
                self.block_size = int(split[1])
                self.iterations = int(split[2])
            except:
                self.block_size = None
                self.iterations = None

    @staticmethod
    def check_valid(data):
        try:
            split = data.key_message.split("/")
            block_size = int(split[1])
            iterations = int(split[2])
        except:
            block_size = None
            iterations = None
        return block_size, iterations

    def is_valid(self):
        if self.has_exif == False or self.block_size == None or self.iterations == None:
            return False
        return True
    
    def set_etif(self, iterations, block_size):
        self.block_size = block_size
        self.iterations = iterations
        self.key_message = "TPE/{0}/{1}".format(self.iterations, self.block_size) # 格式是 TPE/iterations/block_size
        print(self.key_message)
        self._exif['model'] = self.key_message
        print(self._exif['model'])

    def save(self, filename=''):
        name = filename if filename else self._filename
        with open(name, 'wb') as new_image_file:
            new_image_file.write(self._exif.get_file())
        
         


if __name__ == '__main__':
    iterations = 20
    blocksize = 64
    print(f'block_size: {blocksize}, iterations: {iterations}')

    root = Tk()
    root.withdraw()

    filename = askopenfilename()
    img = plt.imread(filename, 0)
    img = img.copy()

    # filter the alpha channel
    if img.shape[2] > 3:
        img = img[:, :, :3]

    # Original
    plot(img, "Original Image")
    #thumbImg = thumbnail(img, blocksize, 'Original thumbnail Image')

    # Encrypt
    enImg = encrypt(img, iterations, blocksize)
    #thumbImg = thumbnail(enImg, blocksize, 'Encrypt thumbnail Image')

    # Decrypt
    _ = decrypt(enImg, iterations, blocksize)

    plt.show()