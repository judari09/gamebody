import cv2
import mediapipe as mp
import numpy as np
from math import acos,degrees,dist
from pynput.keyboard import Key,Controller
import os
img1 = cv2.imread('My project-1.01.png')
img2 = cv2.imread('My project-2.02.png')
img3 = cv2.imread('My project-3.03.png')
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mp_draw = mp.solutions.drawing_utils
mp_holistic  = mp.solutions.holistic
mp_hand = mp.solutions.hands
color_pose_pointer=(10,0,255)
KEYBOARD = Controller()
xc = 320-30
yc = 240-30
xcf = 320+30
ycf = 240+30
xp = 0
yp = 0
xp2 = 0
yp2 = 0
mode= 0
conf = 0
def control1():
    with mp_holistic.Holistic(static_image_mode = False, model_complexity= 1) as holistic:
        with mp_hand.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
            while True:
                ret, frame  = video.read()
                frame = cv2.flip(frame, 1)
                vidrgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(vidrgb)
                resultm=hands.process(vidrgb)
                H, W, _ = frame.shape
                if ret == False:
                    break
                if resultm.multi_hand_landmarks is not None:
                    for hand_landmarks in resultm.multi_hand_landmarks:
                        xp = int(hand_landmarks.landmark[9].x * W)
                        yp = int(hand_landmarks.landmark[9].y * H)
                        xp2= int(hand_landmarks.landmark[8].x * W)
                        yp2= int(hand_landmarks.landmark[8].y * H)
                        xp3 = int(hand_landmarks.landmark[12].x * W)
                        yp3 = int(hand_landmarks.landmark[12].y * H)
                        cv2.circle(frame,(xp,yp),5, color_pose_pointer,3)
                        cv2.circle(frame, (xp, yp), 2, color_pose_pointer, -1)
                        cv2.circle(frame, (xp2, yp2), 5, color_pose_pointer, 3)
                        cv2.circle(frame, (xp2, yp2), 2, color_pose_pointer, -1)
                        cv2.circle(frame, (xp3, yp3), 5, color_pose_pointer, 3)
                        cv2.circle(frame, (xp3, yp3), 2, color_pose_pointer, -1)
                        if yp2 < yp:
                            KEYBOARD.release("z")
                        if yp3 < yp:
                            KEYBOARD.release("x")
                        if yp2 > yp:
                            KEYBOARD.press("z")
                        if yp3 > yp:
                            KEYBOARD.press("x")
                if result.pose_landmarks is not None:
                    x1 = int(result.pose_landmarks.landmark[12].x * W)
                    y1 = int(result.pose_landmarks.landmark[12].y * H)
                    x2 = int(result.pose_landmarks.landmark[11].x * W)
                    y2 = int(result.pose_landmarks.landmark[11].y * H)
                    x3 = int(x1+((x2-x1)/2))
                    y3 = int((y1+y2)/2)
                    cv2.circle(frame, (x3, y3), 5, color_pose_pointer, 3)
                    cv2.circle(frame, (x3, y3), 2, color_pose_pointer, -1)
                    if x3 > xc and x3 < xcf and y3 > yc and y3 < ycf:
                        KEYBOARD.release(Key.up)
                        KEYBOARD.release(Key.down)
                        KEYBOARD.release(Key.left)
                        KEYBOARD.release(Key.right)

                    elif x3 < xc and x3 > 10:
                        KEYBOARD.press(Key.left)
                    elif x3 > xcf and x3 < W-10:
                        KEYBOARD.press(Key.right)
                    elif y3 < yc and y3 > 10:
                        KEYBOARD.press(Key.up)
                    elif y3 > ycf and y3 < H-10:
                        KEYBOARD.press(Key.down)
                    else:
                        cv2.imshow('vid', frame)
                cv2.rectangle(frame, (10,10),(W-10,H-10),(255,0,0),1)
                cv2.rectangle(frame, (10, int(H/2)), (W-10, H-10), (255, 0, 0), 1)
                cv2.rectangle(frame, (10, 10), (int(W/2), H-10), (255, 0, 0), 1)
                cv2.rectangle(frame, (xc,yc),(xcf,ycf),(0,255,0),1)      
                cv2.imshow('vid',frame)    
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break

    video.release()
    cv2.destroyAllWindows()


def control2():
    def orientacion(num):
        if num == 1:

            x0 = int(results.right_hand_landmarks.landmark[0].x * width)
            y0 = int(results.right_hand_landmarks.landmark[0].y * height)

            x9 = int(results.right_hand_landmarks.landmark[9].x * width)
            y9 = int(results.right_hand_landmarks.landmark[9].y * height)

            # cv2.circle(frame, (x0, y0), 6, (0, 255, 255), 4)
            # cv2.circle(frame, (x9, y9), 6, (0, 255, 255), 4)

            if abs(x9 - x0) < 0.05:  # since tan(0) --> ∞
                m = 1000000000
            else:
                m = abs((y9 - y0) / (x9 - x0))

            if m >= 0 and m <= 1:
                if x9 > x0:
                    # cv2.putText(frame, str('>'), (10, 50), 1, 3, (135, 60, 51), 1)
                    #RIGHT
                    return 1

                else:
                    # cv2.putText(frame, str('<'), (10, 50), 1, 3, (135, 60, 51), 1)
                    #LEFT
                    return 0
            if m > 1:
                if y9 < y0:  # since, y decreases upwards
                    # cv2.putText(frame, str('^'), (10, 50), 1, 3, (135, 60, 51), 1)
                    #UP
                    return 2
                else:
                    # cv2.putText(frame, str('v'), (10, 50), 1, 3, (135, 60, 51), 1)
                    #DOWN
                    return 3
        if num == 0:

            x0 = int(results.left_hand_landmarks.landmark[0].x * width)
            y0 = int(results.left_hand_landmarks.landmark[0].y * height)

            x9 = int(results.left_hand_landmarks.landmark[9].x * width)
            y9 = int(results.left_hand_landmarks.landmark[9].y * height)

            # cv2.circle(frame, (x0, y0), 6, (0, 255, 255), 4)
            # cv2.circle(frame, (x9, y9), 6, (0, 255, 255), 4)

            if abs(x9 - x0) < 0.05:  # since tan(0) --> ∞
                m = 1000000000
            else:
                m = abs((y9 - y0) / (x9 - x0))

            if m >= 0 and m <= 1:
                if x9 > x0:
                    # cv2.putText(frame, str('>'), (600, 50), 1, 3, (135, 60, 51), 1)
                    #RIGHT
                    return 1

                else:
                    # cv2.putText(frame, str('<'), (600, 50), 1, 3, (135, 60, 51), 1)
                    #LEFT
                    return 0
            if m > 1:
                if y9 < y0:  # since, y decreases upwards
                    # cv2.putText(frame, str('^'), (600, 50), 1, 3, (135, 60, 51), 1)
                    #UP
                    return 2
                else:
                    # cv2.putText(frame, str('v'), (600, 50), 1, 3, (135, 60, 51), 1)
                    #DOWN
                    return 3
    def cerrado(num):
        if num == 1:

            try:
                x0 = int(results.right_hand_landmarks.landmark[0].x * width)
                y0 = int(results.right_hand_landmarks.landmark[0].y * height)

                x8 = int(results.right_hand_landmarks.landmark[8].x * width)
                y8 = int(results.right_hand_landmarks.landmark[8].y * height)
                d08 = dist([x0, y0], [x8, y8])

                x5 = int(results.right_hand_landmarks.landmark[5].x * width)
                y5 = int(results.right_hand_landmarks.landmark[5].y * height)
                d05 = dist([x0, y0], [x5, y5])

                x12 = int(results.right_hand_landmarks.landmark[12].x * width)
                y12 = int(results.right_hand_landmarks.landmark[12].y * height)
                d012 = dist([x0, y0], [x12, y12])

                x9 = int(results.right_hand_landmarks.landmark[9].x * width)
                y9 = int(results.right_hand_landmarks.landmark[9].y * height)
                d09 = dist([x0, y0], [x9, y9])

                x16 = int(results.right_hand_landmarks.landmark[16].x * width)
                y16 = int(results.right_hand_landmarks.landmark[16].y * height)
                d016 = dist([x0, y0], [x16, y16])

                x13 = int(results.right_hand_landmarks.landmark[13].x * width)
                y13 = int(results.right_hand_landmarks.landmark[13].y * height)
                d013 = dist([x0, y0], [x13, y13])
                if d05 > d08 or d09 > d012 or d013 > d016:
                    # cv2.putText(frame, str('O'), (10, 100), 1, 3, (135, 60, 51), 1)
                    return 1
                else:
                    return 0
            except:
                pass
        elif num == 0:

            try:
                x0 = int(results.left_hand_landmarks.landmark[0].x * width)
                y0 = int(results.left_hand_landmarks.landmark[0].y * height)

                x8 = int(results.left_hand_landmarks.landmark[8].x * width)
                y8 = int(results.left_hand_landmarks.landmark[8].y * height)
                d08 = dist([x0, y0], [x8, y8])

                x5 = int(results.left_hand_landmarks.landmark[5].x * width)
                y5 = int(results.left_hand_landmarks.landmark[5].y * height)
                d05 = dist([x0, y0], [x5, y5])

                x12 = int(results.left_hand_landmarks.landmark[12].x * width)
                y12 = int(results.left_hand_landmarks.landmark[12].y * height)
                d012 = dist([x0, y0], [x12, y12])

                x9 = int(results.left_hand_landmarks.landmark[9].x * width)
                y9 = int(results.left_hand_landmarks.landmark[9].y * height)
                d09 = dist([x0, y0], [x9, y9])

                x16 = int(results.left_hand_landmarks.landmark[16].x * width)
                y16 = int(results.left_hand_landmarks.landmark[16].y * height)
                d016 = dist([x0, y0], [x16, y16])

                x13 = int(results.left_hand_landmarks.landmark[13].x * width)
                y13 = int(results.left_hand_landmarks.landmark[13].y * height)
                d013 = dist([x0, y0], [x13, y13])
                if d05 > d08 or d09 > d012 or d013 > d016:
                    # cv2.putText(frame, str('O'), (600, 100), 1, 3, (135, 60, 51), 1)
                    return 1
                else:
                    return 0
            except:
                pass

    def izquierda(orientacion):
        cv2.waitKey(1)
        KEYBOARD.release(Key.up)
        KEYBOARD.release(Key.down)
        KEYBOARD.release(Key.left)
        KEYBOARD.release(Key.right)
        if orientacion == 0:
            KEYBOARD.press(Key.right)
            #cv2.waitKey(1)
            KEYBOARD.release(Key.up)
        elif orientacion == 1:
            KEYBOARD.press(Key.left)
            #cv2.waitKey(1)
        elif orientacion == 2:
            KEYBOARD.press(Key.up)
            #cv2.waitKey(1)
        elif orientacion == 3:
            KEYBOARD.press(Key.down)
            #cv2.waitKey(1)
        return

    def derecha(orientacion):
        cv2.waitKey(1)
        KEYBOARD.release('z')
        KEYBOARD.release('x')
        #if orientacion == 0:
            #KEYBOARD.press(Key.right)
        #elif orientacion == 1:
            #KEYBOARD.press(Key.left)
        if orientacion == 2:
            KEYBOARD.press('z')
            #cv2.waitKey(80)
            #time.sleep(0.5)

        elif orientacion == 3:
            KEYBOARD.press('z')
            #cv2.waitKey(80)
        return


    mp_drawing = mp.solutions.drawing_utils
    mp_holistc = mp.solutions.holistic

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with mp_holistc.Holistic(
        static_image_mode=False,
        model_complexity=1  ) as holictic:
        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holictic.process(frame_rgb)
            height, width, _ = frame.shape

            if results.right_hand_landmarks is not None:

                dt= cerrado(1)
                if dt == 1:
                    # print('cerrado')
                    df = derecha(5)
                elif dt == 0:
                    ds = orientacion(1)
                    df= derecha(ds)

            if results.left_hand_landmarks is not None:

                dt1 = cerrado(0)
                if dt1 == 1:
                    # print('cerrado')
                    df1 = izquierda(5)
                elif dt1 == 0:
                    ds1 = orientacion(0)
                    df1 = izquierda(ds1)

            frame = cv2.flip(frame, 1)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

def control3():
    with mp_holistic.Holistic(static_image_mode = False, model_complexity= 1) as holistic:
            with mp_hand.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
                while True:
                    ret, frame  = video.read()
                    frame = cv2.flip(frame, 1)
                    vidrgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = holistic.process(vidrgb)
                    resultm=hands.process(vidrgb)
                    H, W, _ = frame.shape
                    if ret == False:
                        break
                    if resultm.multi_hand_landmarks and result.pose_landmarks is not None:
                        for hand_landmarks in resultm.multi_hand_landmarks:
                            x1 = int(result.pose_landmarks.landmark[12].x * W)
                            y1 = int(result.pose_landmarks.landmark[12].y * H)
                            x2 = int(result.pose_landmarks.landmark[11].x * W)
                            y2 = int(result.pose_landmarks.landmark[11].y * H)
                            x3 = int(x1+((x2-x1)/2))
                            y3 = int((y1+y2)/2)
                            xp = int(hand_landmarks.landmark[9].x * W)
                            yp = int(hand_landmarks.landmark[9].y * H)
                            xp2= int(hand_landmarks.landmark[8].x * W)
                            yp2= int(hand_landmarks.landmark[8].y * H)
                            xp3 = int(hand_landmarks.landmark[12].x * W)
                            yp3 = int(hand_landmarks.landmark[12].y * H)
                            xp4 = int(hand_landmarks.landmark[16].x * W)
                            yp4 = int(hand_landmarks.landmark[16].y * H)
                            xp5 = int(hand_landmarks.landmark[20].x * W)
                            yp5 = int(hand_landmarks.landmark[20].y * H)
                        cv2.circle(frame, (x3, y3), 2, (0,0,0), 3)
                        cv2.circle(frame, (x3, y3), 1, (0,0,0), -1)
                        cv2.circle(frame,(xp,yp),2, (255,255,255),3)
                        cv2.circle(frame, (xp, yp), 1, (255,255,255), -1)
                        cv2.circle(frame, (xp2, yp2), 2, (255,255,0), 3)
                        cv2.circle(frame, (xp2, yp2), 1, (255,255,0), -1)
                        cv2.circle(frame, (xp3, yp3), 2, (255,255,0), 3)
                        cv2.circle(frame, (xp3, yp3), 1, (255,255,0), -1)
                        cv2.circle(frame, (xp4, yp4), 2, (255,255,0), 3)
                        cv2.circle(frame, (xp4, yp4), 1, (255,255,0), -1)
                        cv2.circle(frame, (xp5, yp5), 2, (255,255,0), 3)
                        cv2.circle(frame, (xp5, yp5), 1, (255,255,0), -1)
                        cv2.rectangle(frame, (10, 10), (W-10, 150), (255, 0, 255), 1)
                        cv2.rectangle(frame, (10, 151), (206, H-10), (255, 0, 0), 1)
                        cv2.rectangle(frame, (206,151),(413, 315), (255, 0, 0), 1)
                        cv2.rectangle(frame, (206, 316),(413,470 ), (255, 0, 0), 1)
                        cv2.rectangle(frame, (414, 151),(630, H-10), (255, 0, 0), 1)
                        cv2.rectangle(frame, (x3-20, y3-20),(x3+20, y3+20), (255, 0, 0), 1)
                        if xp2 < xp and xp3 < xp and xp4 < xp and xp5 < xp:
                            KEYBOARD.release(Key.up)
                            KEYBOARD.release(Key.down)
                            KEYBOARD.release(Key.left)
                            KEYBOARD.release(Key.right)
                            KEYBOARD.release("z")
                            KEYBOARD.release("x")
                        elif yp2 < yp and yp3 < yp and yp4 < yp and yp5 < yp:
                            KEYBOARD.release(Key.up)
                            KEYBOARD.release(Key.down)
                            KEYBOARD.release(Key.left)
                            KEYBOARD.release(Key.right)
                            KEYBOARD.release("z")
                            KEYBOARD.release("x")
                        elif y3 < 151:
                            if xp2 > xp or xp3 > xp or xp4 > xp or xp5 > xp:
                                KEYBOARD.press("z")
                        elif y3 < 151:
                            if yp2 > yp or yp3 > yp or yp4 > yp or yp5 > yp:
                                KEYBOARD.press("z")
                        elif xp > x3-20:
                            if xp < x3+20 and yp > y3-20 and yp < y3 + 20:
                                KEYBOARD.press("x")
                        if x3 > 414:
                            if yp2 > yp or yp3 > yp or yp4 > yp or yp5 > yp:
                                KEYBOARD.press(Key.right)
                        elif x3 > 414:
                            if xp2 > xp or xp3 > xp or xp4 > xp or xp5 > xp:
                                KEYBOARD.press(Key.right)
                        elif x3 > 10 and x3 < 206:
                            if yp2 > yp or yp3 > yp or yp4 > yp or yp5 > yp:
                                KEYBOARD.press(Key.left)
                        elif x3 > 10 and x3 < 206:
                            if xp2 > xp or xp3 > xp or xp4 > xp or xp5 > xp:
                                KEYBOARD.press(Key.left)
                        elif x3 > 206 and x3 < 414 and y3 < 315:
                            if yp2 > yp or yp3 > yp or yp4 > yp or yp5 > yp:
                                KEYBOARD.press(Key.up)
                        elif x3 > 206 and x3 < 414 and y3 < 315 and y3 > 151:
                            if xp2 > xp or xp3 > xp or xp4 > xp or xp5 > xp:
                                KEYBOARD.press(Key.up)
                        elif x3 > 206 and x3 < 414 and y3 > 315:
                            if yp2 > yp or yp3 > yp or yp4 > yp or yp5 > yp:
                                KEYBOARD.press(Key.down)
                        elif x3 > 206 and x3 < 414 and y3 > 315:
                            if xp2 > xp or xp3 > xp or xp4 > xp or xp5 > xp:
                                KEYBOARD.press(Key.down)

                    cv2.imshow('vid',frame)    
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        break
            video.release()
            cv2.destroyAllWindows()
 
with mp_hand.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
    while True:
        ret, frame = video.read()
        frame = cv2.flip(frame, 1)
        vidrgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultm = hands.process(vidrgb)
        H, W, _ = frame.shape
        if ret == False:
            break
        if resultm.multi_hand_landmarks is not None:
            for hand_landmarks in resultm.multi_hand_landmarks:
                xp = int(hand_landmarks.landmark[9].x * W)
                yp = int(hand_landmarks.landmark[9].y * H)
                xp2 = int(hand_landmarks.landmark[8].x * W)
                yp2 = int(hand_landmarks.landmark[8].y * H)
                xp3 = int(hand_landmarks.landmark[12].x * W)
                yp3 = int(hand_landmarks.landmark[12].y * H)
                xp4 = int(hand_landmarks.landmark[16].x * W)
                yp4 = int(hand_landmarks.landmark[16].y * H)
                xp5 = int(hand_landmarks.landmark[20].x * W)
                yp5 = int(hand_landmarks.landmark[20].y * H)
                cv2.circle(frame, (xp, yp), 2, (255, 255, 255), 3)
                cv2.circle(frame, (xp, yp), 1, (255, 255, 255), -1)
                cv2.circle(frame, (xp2, yp2), 2, (255, 255, 0), 3)
                cv2.circle(frame, (xp2, yp2), 1, (255, 255, 0), -1)
                cv2.circle(frame, (xp3, yp3), 2, (255, 255, 0), 3)
                cv2.circle(frame, (xp3, yp3), 1, (255, 255, 0), -1)
                cv2.circle(frame, (xp4, yp4), 2, (255, 255, 0), 3)
                cv2.circle(frame, (xp4, yp4), 1, (255, 255, 0), -1)
                cv2.circle(frame, (xp5, yp5), 2, (255, 255, 0), 3)
                cv2.circle(frame, (xp5, yp5), 1, (255, 255, 0), -1)
                if yp2<yp and yp3>yp and yp4>yp and yp5>yp:
                    cv2.imshow('vid', frame)
                    cv2.imshow('control',img1)
                    mode=1
                elif yp2 < yp and yp3 < yp and yp4 > yp and yp5 > yp:
                    cv2.imshow('vid', frame)
                    cv2.imshow('control', img2)
                    mode=2
                elif yp2 < yp and yp3 < yp and yp4 < yp and yp5 > yp:
                    cv2.imshow('vid', frame)
                    cv2.imshow('control', img3)
                    mode=3
                else:
                    cv2.imshow('vid', frame)
                if mode == 1 and yp2 > yp and yp3 > yp and yp4 > yp and yp5 > yp:
                    conf=1
                    cv2.destroyAllWindows()
                elif mode == 2 and yp2 > yp and yp3 > yp and yp4 > yp and yp5 > yp:
                    conf = 2
                    cv2.destroyAllWindows()
                elif mode == 3 and yp2 > yp and yp3 > yp and yp4 > yp and yp5 > yp:
                    conf = 3
                    cv2.destroyAllWindows()
                while conf==1:
                    os.chdir("NES-copia")
                    os.system("nestopia.exe")
                    control1()
                while conf == 2:
                    os.chdir("NES-copia")
                    os.system("nestopia.exe")
                    control2()
                while conf == 3:
                    os.chdir("NES-copia")
                    os.system("nestopia.exe")
                    control3()
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
video.release()
cv2.destroyAllWindows()

