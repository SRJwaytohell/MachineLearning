# MachineLearning#from theano.tensor.signal import downsample
#import deepnet as dn
import cv2
#print(cv2.__version__)
#cap.release()
cap = cv2.VideoCapture('C:\\Users\\JARVIS\\Desktop\\Jupiter\\Tom_and_Jerry\\Dataset\\Train_Tom_and_Jerry.mp4')
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #print(gray.shape)
    oneDarray = gray.flatten()
    #print(oneDarray.shape)

    _,threshold = cv2.threshold(gray,250,255,0)
    
    #cv2.rectangle(gray,(384,0),(510,128),(255,255,0),3)


    #####detect contours
    contours,hier = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(frame,contours,-1,(255,0,255),3)

    #####show image
    if ret == True:
        cv2.imshow('Frame',frame)
        #cv2.imshow('threshold',threshold)
        #cv2.imshow('gray',gray)
        cv2.imshow('img',oneDarray)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
