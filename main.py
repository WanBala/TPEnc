import decrypt
import matplotlib.pyplot as plt

iterations = 20
blocksize = 16

img = plt.imread('good.jpg')

TPE = decrypt.TPEncryption(img)

ans = input('[E]ncrypt or [D]ecrypt? ')
if ans == 'E':
    TPE.encrypt(iterations, blocksize)
elif ans == 'D':
    TPE.decrypt(iterations, blocksize)
else:
    print('No such operation')