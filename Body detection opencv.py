#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


# load the cascade classifier 


# In[7]:


face_cascade=cv2.CascadeClassifier('haarcascade_fullbody.xml')


# In[4]:


# start webcam/or load video


# In[5]:


video=cv2.VideoCapture(r'C:\Users\ritesh gupta\Downloads\MILA TOH MAREGA (Warrior Version) _ Ft. Indian Army _ Indian Air Force.mp4')


# In[10]:


while True:
    #read image from video
    respose, color_img=video.read()
    
    if respose==False: #( agar image nhi mila to..)
        break
        
    # convert to grayscale
    gray_img=cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)
    
    # detect the faces
    faces=face_cascade.detectMultiScale(gray_img,1.2,1)
    
    #display rectangle
    for (x,y,w,h) in faces:
        cv2.rectangle(color_img,(x,y),(x+w,y+h),(0,0,255),2)
        
        #display image
        cv2.imshow('img',color_img)
        
    key=cv2.waitKey(1) # this will generate a new frame after every 1millisecond
    if key==ord('q'): # once you enter q the window will be detroyed.
        break
        
#Release the video capture object
video.release()
cv2.destroyAllWindows()
        


# In[ ]:




