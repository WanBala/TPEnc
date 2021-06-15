import decrypt
import encryption
import matplotlib.pyplot as plt

iterations = 1
blocksize = 200

img = plt.imread('lena.png', 0)
plt.figure(1)
plt.imshow(img)
img = img.copy()

ans = input('[E]ncrypt or [D]ecrypt? ')
if ans == 'E':
    encryption.encryption(img, iterations, blocksize)
elif ans == 'D':
    decrypt.decrypt(img, iterations, blocksize)
else:
    print('No such operation')