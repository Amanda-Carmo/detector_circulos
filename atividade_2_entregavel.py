__author__      = "Matheus Dib, Fabio de Miranda"

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import math

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1

print("Press q to QUIT")

cor_menor = np.array([82,  50,  50], dtype=np.uint8)
cor_maior = np.array([112, 255, 255], dtype=np.uint8)

cor_menorm = np.array([125,  50,  50], dtype=np.uint8)
cor_maiorm = np.array([185, 255, 255], dtype=np.uint8)


# Cria o detector BRISK
brisk = cv2.BRISK_create()

# Configura o algoritmo de casamento de features que vÃª *como* o objeto que deve ser encontrado aparece na imagem
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Define o mínimo de pontos similares
MINIMO_SEMELHANCAS = 7


def auto_canny(image, sigma=0.33):

    v = np.median(image)


    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged




def find_good_matches(descriptor_image1, frame_gray):    
        
    des1 = descriptor_image1
    kp2, des2 = brisk.detectAndCompute(frame_gray,None)

    matches = bf.knnMatch(des1,des2,k=2)

    good = []
    
    # Desenha um contorno em vermelho ao redor de onde o objeto foi encontrado
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    return kp2, good


def find_homography_draw_box(kp1, kp2, img_cena):
    
    out = img_cena.copy()
    
    kp2, good = find_good_matches(des1, gray)
    
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img_original.shape
    
    # Um retângulo com as dimensões da imagem original
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    
    if M is not None:

        # Transforma os pontos do retângulo para onde estao na imagem destino usando a homografia encontrada
        dst = cv2.perspectiveTransform(pts,M)

        # Desenha um contorno em vermelho ao redor de onde o objeto foi encontrado 
        img2b = cv2.polylines(out,[np.int32(dst)],True,(255,255,0),5, cv2.LINE_AA)
    
        return img2b    


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    original_rgb = cv2.imread("insper_logo.png")  # Imagem a procurar
    img_original = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2GRAY)

    # Encontra os pontos (keypoints) nas duas imagems
    kp1, des1 = brisk.detectAndCompute(img_original ,None)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray,(5,5),0)
    
        bordas = auto_canny(blur)
        
        circles = []
    
        # Obtains a version of the edges image where we can draw in color
        bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    
        circles = None
        circles=cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=1,maxRadius=100)
        
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
        
        mask = cv2.inRange(img_hsv, cor_menor, cor_maior)  
        maskm = cv2.inRange(img_hsv, cor_menorm, cor_maiorm)
        
        frame_rgb = frame 
        
        framed = None
        
        kp2, good_matches = find_good_matches(des1, gray)
        
        if len(good_matches) > MINIMO_SEMELHANCAS:
            #img3 = cv2.drawMatches(original_rgb,kp1,bordas_color,kp2, good_matches,      None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)           
            
            img3 = cv2.drawMatches(original_rgb,kp1,bordas_color,kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            bordas_color = find_homography_draw_box(kp1, kp2, bordas_color)   
            cv2.imshow('feature points', img3)        
        
        ccx = None
        ccy = None
        cmx = None
        cmy = None  
        
        D = None
        angle = None
    
        if circles is not None:     
            
            circles = np.uint16(np.around(circles))
            
         
            for i in circles[0,:]:              
    
                # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])          
                cv2.circle(bordas_color,(i[0],i[1]),i[2],(0,255,0),2)
                
                # draw the center of the circle
                cv2.circle(bordas_color,(i[0],i[1]),2,(0,0,255),3)
                
                if maskm[i[1]][i[0]] > 240:
                    cv2.circle(bordas_color,(i[0],i[1]),i[2],(153,0,255),-1)     
                    cmx = i[0]
                    cmy = i[1]     
                
                elif mask[i[1]][i[0]] > 240:
                    cv2.circle(bordas_color,(i[0],i[1]),i[2],(255,153,0),-1)
                    ccx = i[0]
                    ccy = i[1]
                    
                if ccx is not None and ccy is not None and cmx is not None and cmy is not None: 
                    cv2.line(bordas_color, (ccx,ccy), (cmx,cmy), (0, 255, 0), 3)
                    di = (((int(ccx) - int(cmx))**2) + ((int(ccy) - int(cmy))**2))**0.5
                    dicm = di * 0.02645833    
                    
                    f = 13
      
                    D = (f*14)/dicm
                    
                    print(D)
                    
                    #angle = np.rad2deg(np.arctan2(int(ccy) - int(cmy), int(ccx) - int(cmx)))
                    angle = np.arctan((ccy - cmy)/(ccx - cmx))*180/np.pi
                    
                    #angle = np.arctan(abs(ccy - cmy), abs(ccx - cmx))*180/math.pi 
                    
        #cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])    
        #cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(bordas_color,'Press q to quit',(1,15), font, 0.3,(255,255,255),1,cv2.LINE_AA)
        
        if D is not None:
            cv2.putText(bordas_color,"Distance from camera: {0}".format(D),(1,35), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        
        
        if angle is not None:
            cv2.putText(bordas_color,"Angle: {0}".format(angle),(1,50), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        
        # Display the resulting frame
        #cv2.imshow('Detector de circulos',bordas_color)
        cv2.putText(bordas_color,"Distance from camera: ",(1,35), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(bordas_color,"Angle: ",(1,50), font, 0.5,(255,255,255),1,cv2.LINE_AA)        
        
        frame_rgb = frame 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        

        cv2.imshow('Detector de circulos', bordas_color)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    #  When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
