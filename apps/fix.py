#!/usr/bin/env python
# coding: utf-8

# In[1]:


class fix:
    def __init__(self, img):
        self.img = img

    # BGR에서 HSV로 변환해서, red_range 부분만을 filtering
    def hsv(self):
        import numpy as np
        import cv2
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)       
        lower = np.array([0,100,100])      # lower 값을 0, 50, 50으로 주면 훨씬 깔끔하게 나오지만, 1번의 함.png같은 경우는 ㅎ이 위아래가 붙음 그래서 100이 맞음
        upper = np.array([179,255,255])                     
        mask = cv2.inRange(img, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)
        return result

    # gray_scale을 통해 binary로 변환한 2차원 이미지 데이터의 픽셀 강도를 선명하게 바꾸는 스크래치 작업
    def sunmyung(self, img):
        import numpy as np
        a = []
        for i in img:
            b = []
            for j in i:
                if j < 80:
                    b.append(j)
                elif j >= 80 and j <= 255:
                    b.append(255)
                else:
                    b.append(j)
        
            a.append(b) 
        a = np.array(a).astype('uint8')
        return a

    # 이미지의 윤곽선을 검출하는 cv2.contours를 사용, drawing을 통해, 일정크기의 윤곽선만 검출하여 그 주변을 cv2.circle로 보정을한 후, 색을 칠하기(일부노이즈제거)
    def Contour(self):
        import numpy as np
        import cv2
        # findContours를 하기 위한 이진화 변수 설정
        red = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) 
        red = np.reshape(red, (red.shape[0], red.shape[1], 1))
        red = cv2.threshold(red,127,255, cv2.THRESH_BINARY)[1]
        red = np.reshape(red, (red.shape[0], red.shape[1], 1))
        
        # findContours에서 일정 길이를 가진 윤곽선만 추출
        def drawing(contours):
            a = []
            for i in contours:
                if len(i) > 50:      # 50 -> 25으로 변경.
                    a.append(i)
            
            a = tuple(a)
            return a
            
        contours, _ = cv2.findContours(red, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # 일정 길이 윤곽선만 detection
        contours = drawing(contours)
        
        # 해당 윤곽선에 circle 라인 그리기.
        # for i in range(len(contours)):
        #     for j in range(len(contours[i])):                                # circle 두께
        #         cv2.circle(self.img, (contours[i][j][0][0], contours[i][j][0][1]), 1, (0, 0, 255), -1)
        
        
        # 3번째 인자가 음수면 모든 contour를 그림
        # 빨간색 contour 선
        # 선의 두께는 1 pixel(음수값으로 주면, 내부가 채워진다.)
        # 윤곽선 내부 색 채우기.
        cv2.drawContours(self.img, contours, -1, (0, 0, 255), -1)
        return self.img

    # threshold를 통해, 이미지를 gray_scale 후, 흑백 binary로 변환
    def threshold(self):
        import numpy as np
        import cv2
        img = cv2.cvtColor(self.img, cv2.COLOR_HSV2BGR)                              
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                        
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        img = cv2.threshold(img,127,255, cv2.THRESH_BINARY)[1]            
        # img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        return img

    # 이미지 denoising 작업 + erode와 dilate를 통해 일부 노이즈 제거 후, 복원 + 픽셀 sunmyung화 작업
    def Noise(self):
        import numpy as np
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        img = cv2.fastNlMeansDenoising(self.img,None,80,7,21)
        img = cv2.erode(img, k, iterations = 1)
        img = cv2.dilate(img, k, iterations = 1)
        img = self.sunmyung(img)
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        return img

