#!/usr/bin/env python
# coding: utf-8

# In[1]:


class metrics:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
    # ------------------------------------------------------------------------------------------
    def cosine_similarity(self, img1, img2):
        import numpy as np
        array1 = img1
        array2 = img2
        assert array1.shape == array2.shape
        
        h, w = array1.shape
        len_vec = h * w
        vector_1 = array1.reshape(len_vec,) / 255.
        vector_2 = array2.reshape(len_vec,) / 255.
    
        cosine_similarity = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
        return cosine_similarity
    
    # ------------------------------------------------------------------------------------------
    def ssim_similarity(self, img1, img2):
        from skimage.metrics import structural_similarity as ssim    # 1에 가까울수록 유사
        import numpy as np
        grayA = img1
        grayB = img2
        
        (score, diff) = ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        return score
    
    # ------------------------------------------------------------------------------------------
    # 기본값, erode 1번, erode 2번, dilate 1번, dilate 2번 값이 저장.
    
    def rank(self):
        import numpy as np
        import cv2
        from apps.registration import get_registration, calc_mse
        fixed = self.img1
        moving = self.img2
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mse_rank = []
        cos_rank = []
        ss_rank = []
        mse_rank.append(calc_mse(fixed, moving))
        cos_rank.append(self.cosine_similarity(fixed, moving))
        ss_rank.append(self.ssim_similarity(fixed, moving))
        for kk in range(2):
            kk += 1
            mse_rank.append(calc_mse(cv2.erode(fixed, k, iterations = kk), moving))
            cos_rank.append(self.cosine_similarity(cv2.erode(fixed, k, iterations = kk), moving))
            ss_rank.append(self.ssim_similarity(cv2.erode(fixed, k, iterations = kk), moving))
            mse_rank.append(calc_mse(cv2.dilate(fixed, k, iterations = kk), moving))
            cos_rank.append(self.cosine_similarity(cv2.dilate(fixed, k, iterations = kk), moving))
            ss_rank.append(self.ssim_similarity(cv2.dilate(fixed, k, iterations = kk), moving))
    
        return min(mse_rank), max(cos_rank), max(ss_rank)

