%matplotlib inline
import matplotlib.pyplot as plt

dictVal={
    'p1':{'photo':[[350,200],[460,300]],
          'name':[[40,105],[170,125]],
          'surname':[[40,140],[140,160]],
          'dob':[[40,180],[130,200]],
          'panNumber':[[40,215],[150,240]]
         },
    'p2':{'photo':[[540,300],[680,410]],
          'name':[[40,150],[380,180]],
          'surname':[[40,205],[380,240]],
          'dob':[[40,255],[180,290]],
          'panNumber':[[40,320],[220,350]]
         },
    'p3':{'photo':[[225,110],[275,170]],
          'name':[[20,60],[125,80]],
          'surname':[[20,85],[130,100]],
          'dob':[[20,105],[85,118]],
          'panNumber':[[20,130],[100,150]]
         },
    'p5':{'photo':[[400,260],[510,370]],
          'name':[[50,140],[200,170]],
          'surname':[[50,180],[220,210]],
          'dob':[[50,220],[160,250]],
          'panNumber':[[50,270],[180,290]]
         }
}
import cv2
curImage=plt.imread('./panCard/p5.jpeg')
print(curImage.shape)
curImage=cv2.rectangle(curImage,(400,260),(510,370),(255,0,0),2)
curImage=cv2.rectangle(curImage,(50,140),(200,170),(255,0,0),2)
curImage=cv2.rectangle(curImage,(50,180),(220,210),(255,0,0),2)
curImage=cv2.rectangle(curImage,(50,220),(160,250),(255,0,0),2)
curImage=cv2.rectangle(curImage,(50,270),(180,290),(255,0,0),2)
plt.imshow(curImage)
plt.show()

# We will perform the following operations
%matplotlib inline
import cv2
import matplotlib.pyplot as plt

# Add background padding single color

curImage=plt.imread('./panCard/p5.jpeg')

def addPaddingWithSingleColor(curImage,color,top,bottom,left,right,dictVal):
    alteredImage=cv2.copyMakeBorder(curImage.copy(),top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLUE)
    photoVal=dictVal['photo']
    photoVal=[[photoVal[0][0] + left,photoVal[0][1] + top],[photoVal[1][0] + left,photoVal[1][1] + top]]
    nameVal=dictVal['name']
    nameVal=[[nameVal[0][0] + left,nameVal[0][1] + top],[nameVal[1][0] + left,nameVal[1][1] + top]]
    surnameVal=dictVal['surname']
    surnameVal=[[surnameVal[0][0] + left,surnameVal[0][1] + top],[surnameVal[1][0] + left,surnameVal[1][1] + top]]
    dobVal=dictVal['dob']
    dobVal=[[dobVal[0][0] + left,dobVal[0][1] + top],[dobVal[1][0] + left,dobVal[1][1] + top]]
    panNumberVal=dictVal['panNumber']
    panNumberVal=[[panNumberVal[0][0] + left,panNumberVal[0][1] + top],[panNumberVal[1][0] + left,panNumberVal[1][1] + top]]
    return([alteredImage,{'photo':photoVal,'name':nameVal,'surName':surnameVal,'dob':dobVal,'panNumber':panNumberVal}])

def addPaddingWithImage(curImage,paddingImage,top,bottom,left,right,dictVal):
    curImageWidth=curImage.shape[0]
    curImageHeight=curImage.shape[1]
    alteredImageWidth=curImageWidth + left+right
    alteredImageHeight=curImageHeight +top+bottom
    alteredImage=cv2.resize(paddingImage,(alteredImageHeight,alteredImageWidth))
    print(curImage.shape)
    print(curImageWidth)
    print(curImageHeight)
    print(alteredImage.shape)
    alteredImage[left:left+curImageWidth,top:top+curImageHeight,:]=curImage[:,:,:]
    photoVal=dictVal['photo']
    photoVal=[[photoVal[0][0] + top,photoVal[0][1] + left],[photoVal[1][0] + top,photoVal[1][1] + left]]
    nameVal=dictVal['name']
    nameVal=[[nameVal[0][0] + top,nameVal[0][1] + left],[nameVal[1][0] + top,nameVal[1][1] + left]]
    surnameVal=dictVal['surname']
    surnameVal=[[surnameVal[0][0] + top,surnameVal[0][1] + left],[surnameVal[1][0] + top,surnameVal[1][1] + left]]
    dobVal=dictVal['dob']
    dobVal=[[dobVal[0][0] + top,dobVal[0][1] + left],[dobVal[1][0] + top,dobVal[1][1] + left]]
    panNumberVal=dictVal['panNumber']
    panNumberVal=[[panNumberVal[0][0] + left,panNumberVal[0][1] + top],[panNumberVal[1][0] + left,panNumberVal[1][1] + top]]
    return([alteredImage,{'photo':photoVal,'name':nameVal,'surName':surnameVal,'dob':dobVal,'panNumber':panNumberVal}])



BLUE = [255,255,0]
top=10
bottom=10
left=70
right=50
image1,dictVal1=addPaddingWithSingleColor(curImage,[255,255,0],top,bottom,left,right,dictVal['p5'])
plt.imshow(cv2.rectangle(image1,(dictVal1['photo'][0][0],dictVal1['photo'][0][1]),(dictVal1['photo'][1][0],dictVal1['photo'][1][1]),(255,0,0),2))
plt.show()
plt.figure()
paddingImage=plt.imread('./panCard/{0}'.format('pattern1.jpg'))
image1,dictVal1=addPaddingWithImage(curImage,paddingImage,top,bottom,left,right,dictVal['p5'])
print(dictVal1)
plt.imshow(cv2.rectangle(image1,(dictVal1['photo'][0][0],dictVal1['photo'][0][1]),(dictVal1['photo'][1][0],dictVal1['photo'][1][1]),(255,0,0),2))
