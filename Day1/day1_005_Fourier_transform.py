#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on 2021

@author: H.J
"""


# In[2]:


# =============================================================================
# # Fourier Transform
# 
# Numpy에서 FFT함수
# cv2.dft(), cv2.idft()
# =============================================================================

# 출처: https://leechamin.tistory.com/266 [참신러닝 (Fresh-Learning)]

# =============================================================================
# Fourier Transform은 다양한 필터의 주파수(frequency) 특성을 분석하는 데 사용
# 이미지의 경우, 주파수 영역을 찾기 위해 2D Discrete Fourier Transfrom(DFT)를 사용
# Fast Fourier Transform(FFT)이라는 고속 알고리즘이 이미지 처리 또는 신호 처리 사용
# 사인곡선의 신호의 경우, x(t)=Asin(2πft)에서 f는 신호의 주파수를 나타냄
# 신호가 샘플링되어서 이산 신호를 형성하면 동일한 주파수 영역을 얻지만, 
# [−π,π] 또는 [0,2π]의 범위에서 주기적 이미지를 두 방향으로 샘플링되는 신호로 간주
# 따라서 Fourier Transform을 X와 Y의 방향으로 하면 이미지의 주파수 얻을 수 있음
# 사인곡선의 신호의 경우, 짧은 시간에 진폭이 그렇게 빠르게 변화하면 고주파 신호로,
# 천천히 변화하면 저주파 신호임
# 같은 아이디어를 이미지로 확장 가능
# 이미지에서 진폭이 크게 변하는 곳은 가장자리 부분이나 노이즈가 있는 부분
# 가장자리와 노이즈는 이미지에서 고주파 부분이라고 할 수 있음
# 진폭에 큰 변화가 없으면 저주파 성분임
# =============================================================================


# In[3]:


# =============================================================================
# [1] np.fft.fft2() : 복잡한 배열로 주파수 변환

# 사용법: np.fft.fft2(흑백이미지, 배열 크기)
# - 배열 크기가 입력 이미지의 크기보다 크면 입력 이미지는 FFT 연산이전에 제로-패딩 진행
# - 입력 이미지보다 작을 경우, 입력 이미지는 잘림(cropped)
# - 아무 인자도 안넘겨진다면, 결과 배열의 크기는 입력과 동일
# =============================================================================

# 0 주파수 성분은 왼쪽 위에 위치
# 이를 가운데로 이동하고 싶으면, 두 방향으로 N/2만큼 결과를 옮겨야 함
# - np.fft.fftshift()로 가능
# 주파수 변환을 찾으면 크기 스펙트럼도 찾을 수 있음


import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('./img/boy_face.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.title("Input Image")
plt.axis('off')
plt.subplot(122)
plt.imshow(magnitude_spectrum,cmap='gray')
plt.title("Magnitude Spectrum")
plt.axis('off')
plt.show()

# 가운데의 흰색 영역은 저주파수 함량이 더 많다는 것을 의미함


# In[4]:


# =============================================================================
# 주파수 변환을 찾았으니 이제 고역 통과 필터링, 이미지 재구성, 
# 역(inverse) DFT와 같은 주파수 영역의 기능
# 이를 위해 60x60 크기의 직사각형 창으로 마스킹해서 낮은 주파수를 제거
# np.fft.ifftshift() 함수를 사용하여 역방향 이동 적용하여 
# DC 성분이 왼쪽 상단에 오도록 변경
# 그런 다음 np.ifft2() 함수를 사용하여 역 FFT를 찾음 -> 절대값을 적용
# =============================================================================


rows,cols = img.shape 
crow,ccol = round(rows/2), round(cols/2) 
 # 소수점으로 떨어지는 것을 방지하기 위함 : round 

# 60x60 크기로 창 만들기 
fshift[crow-30:crow+60, ccol-30:ccol+30] = 0 
plt.figure(figsize=(12,8)) 
plt.imshow(np.abs(fshift)) 


f_ishift = np.fft.ifftshift(fshift) 
plt.figure(figsize=(12,8)) 
plt.imshow(np.abs(f_ishift)) 

img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back) 

plt.figure(figsize=(12,8)) 
plt.subplot(131),plt.imshow(img,cmap='gray') 
plt.title("Input Image"), plt.axis('off') 
plt.subplot(132),plt.imshow(img_back,cmap='gray') 
plt.title("Image after HPF"), plt.axis('off') 
plt.subplot(133),plt.imshow(img_back) 
plt.title("Result in JET"), plt.axis('off') 
plt.show()


img_back.max()

# HPF(High Pass Filtering)가 가장자리 검출 역할 -> Image Gradient
# 대부분의 이미지 데이터가 스펙트럼의 저주파 영역에 존재


# In[5]:


# =============================================================================
# (2) OpenCV : cv2.dft()와 cv2.idft()
# =============================================================================

img = cv2.imread('./img/boy_face.jpg',0)
dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
dft.shape
plt.imshow(np.abs(dft[:,:,1]))

# 원점 이동
dft_shift = np.fft.fftshift(dft) 
dft_shift.shape
plt.imshow(np.abs(dft_shift[:,:,0]))


magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
magnitude_spectrum.shape

plt.subplot(121), plt.imshow(img,cmap='gray')
plt.title("Input Image"), plt.axis('off') 
plt.subplot(122), plt.imshow(magnitude_spectrum,cmap='gray') 
plt.title("Magnitude Spectrum"), plt.axis('off') 
plt.show()


# 한 번에 크기와 위상을 반환하는 cv2.cartToPolar()를 사용 가능


# In[6]:


# =============================================================================
# 이미지에서 고주파 부분 제거
# - 저주파 : 높은값(1)로 마스크, 고주파 : 0  
# =============================================================================

rows, cols = img.shape
crow,ccol = round(rows/2), round(cols/2)

# 마스크를 먼저 생성하고, 가운데 네모를 1로 나머지를 0
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30,ccol-30:ccol+30] = 1

plt.imshow(mask[:,:,0])

# 마스크와 역 DFT를 적용
fshift = dft_shift*mask

plt.imshow(fshift[:,:,0])

f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.figure(figsize=(12,8))
plt.subplot(121), plt.imshow(img,cmap='gray')
plt.title("Input Image"), plt.axis('off')
plt.subplot(122), plt.imshow(img_back,cmap='gray')
plt.title("Magnitude Spectrum"), plt.axis('off')
plt.show()


# In[7]:


# =============================================================================
# Performance Optimization of DFT
# =============================================================================

# 보통, OpenCV의 cv2.dft()와 cv2.idft() 함수는 Numpy 보다 빠름
# 하지만 Numpy 함수는 좀 더 유저친화적임
 

# DFT 연산의 성능은 일부 배열 크기에서 더 좋음
# 이는 배열의 크기가 2배 일 때 더 빠름. 배열의 크기가 2배, 3배, 5배인 경우도 효율적
# 코드의 성능에 대해 걱정이라면, 배열의 크기를 DFT를 찾기 이전에
# 어떠한 최적의 크기(제로-패딩으로)로 수정하면 됨
# OpenCV에서는, 직접 제로패딩을 해야함
# Numpy에서는, FFT 계산의 새 크기를 지정하면 자동으로 0으로 패딩됨

# OpenCV는 cv2.getOptimalDFTSize()를 통해 최적의 크기 찾을 수 있음
# cv2.dft()와 np.fft.fft2() 모두에 적용


img = cv2.imread('./img/boy_face.jpg',0)
rows,cols = img.shape
print(rows,cols)
# 501 398

nrows = cv2.getOptimalDFTSize(rows)
ncols = cv2.getOptimalDFTSize(cols)
print(nrows,ncols)
# 512 400

# 0 패딩 생성
# - 커다란 0 배열을 만들어도 되고
# - 데이터를 복사하고 cv2.copyMakeBorder()를 사용해도 됨

nimg = np.zeros((nrows,ncols))
nimg[:rows,:cols] = img


right = ncols - cols
bottom = nrows - rows
bordertype = cv2.BORDER_CONSTANT
nimg = cv2.copyMakeBorder(img,0,bottom,0,right,bordertype, value=0)


# In[8]:


# =============================================================================
# >>>%timeit fft1 = np.fft.fft2(img)
# 71.4 ms ± 1.62 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# >>>%timeit fft2 = np.fft.fft2(img,[nrows,ncols])
# 8.1 ms ± 330 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# ==> 이는 4배 빠름
# 
# >>>%timeit dft1= cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
# 2.13 ms ± 101 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# >>>%timeit dft2= cv2.dft(np.float32(nimg),flags=cv2.DFT_COMPLEX_OUTPUT)
# 1.16 ms ± 69.4 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# ==> 4배 빠름, OpenCV 함수가 Numpy 함수보다 3배정도 빠름, 역 FFT도 비슷
# =============================================================================


# 조정 파라미터 없는 간단한 평균 필터
mean_filter = np.ones((3,3))

# 가우시안 필터 만들기
x = cv2.getGaussianKernel(3,3)
gaussian = x*x.T

## 다른 가장자리 검출 필터들
# x 방향으로의 Scharr 
scharr = np.array([[-3,0,3],
                  [-10,0,10],
                  [-3,0,3]])

# x 방향으로의 Sobel
sobel_x = np.array([[-1,0,1],
                    [-2,0,2],
                   [-1,0,1]])

# y 방향으로의 Sobel
sobel_y = np.array([[-1,-2,-1],
                   [0,0,0],
                   [1,2,1]])

# laplacian
laplacian = np.array([[0,1,0],
                     [1,-4,1],
                     [0,1,0]])

filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x',                 'sobel_y', 'scharr_x']

fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]

for i in range(6):
    plt.subplot(2,3,i+1), plt.imshow(mag_spectrum[i],cmap='gray')
    plt.title(filter_name[i]), plt.axis('off')

plt.show()


# 결과 이미지에서, 각 커널이 차단하는 주파수 영역과 통과되는 영역을 확인 할 수 있음 

