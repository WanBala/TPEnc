import decrypt
import matplotlib.pyplot as plt

iterations = 1
blocksize = 700

img = plt.imread('good.jpg')
plt.figure(1)
plt.imshow(img)
img = img.copy()

TPE = decrypt.TPEncryption(img)

ans = input('[E]ncrypt or [D]ecrypt? ')
if ans == 'E':
    TPE.encrypt(iterations, blocksize)
elif ans == 'D':
    TPE.decrypt(iterations, blocksize)
else:
    print('No such operation')