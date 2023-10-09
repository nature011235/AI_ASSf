
import logging
import time
from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from pynput import mouse
from PIL import ImageGrab
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pyautogui
import pydirectinput
windows = pyautogui.getAllWindows()
pyautogui.FAILSAFE=False
# logger = logging.getLogger('TfPoseEstimatorRun')
# logger.handlers.clear()
# logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)
w, h = 656,368
e = TfPoseEstimator(get_graph_path('cmu'), target_size=(w, h))
# def on_click(x, y, button, pressed):
#     if pressed:
#         print(f"Mouse clicked at ({x}, {y}) with button {button}")
#         if button==mouse.Button.middle:
#             img_rgb = ImageGrab.grab()
#             img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
#             # humans = e.inference(img_bgr, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
#             # img_bgr = TfPoseEstimator.draw_humans(img_bgr, humans, imgcopy=False)
            
#             cv2.imshow('result', img_bgr)
       
#         # if button==mouse.Button.middle:
#         #     pyautogui.moveTo(200, 200, duration=0)

# cv2.namedWindow('window', cv2.WINDOW_KEEPRATIO)
# cv2.resizeWindow('window', 1920, 1080)  
# img=np.array([0 for i in range(1920*1080*3)])
# img.reshape(1920,1080,3)
# img = img.astype(np.uint8)
# cv2.imshow('window',img)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#                 exit()
def on_click(x, y, button, pressed):
    if pressed:
        print(f"Mouse clicked at ({x}, {y}) with button {button}")
        if button==mouse.Button.middle:
            img_rgb = ImageGrab.grab(bbox=(640,360,1280,720))
            img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            global w,h,e
            humans = e.inference(img_bgr, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
            img_bgr,location= TfPoseEstimator.draw_humans(img_bgr, humans, imgcopy=False)
            pydirectinput.PAUSE=0.0
            pydirectinput.moveRel((int(location[0]+640))-(int(1920/2)), (int(location[1]+360))-(int(1080/2)), relative=True)
            time.sleep(0.001)
           
            cv2.namedWindow('window', 0)
            cv2.resizeWindow('window', 960, 540)  
            
            cv2.imshow('window', img_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()
        
        #     pyautogui.moveTo(200, 200, duration=0)
if __name__ == '__main__':
    print("loading completed")
    mouse_listener = mouse.Listener(on_click=on_click)
    mouse_listener.start()
    # image = Image.open("C:\\Users\\natur\\aa.jpg")
    # img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # humans = e.inference(img_bgr, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
    # img_bgr = TfPoseEstimator.draw_humans(img_bgr, humans, imgcopy=False)
    # cv2.imshow('aa', img_bgr)

    # e = TfPoseEstimator(get_graph_path('cmu'), target_size=(w, h))

    # # estimate human poses from a single image !
    # image = cv2.imread('C:\\Users\\natur\\aa.jpg')
    # humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
    # image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    


    # fig = plt.figure()
        
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()
    mouse_listener.join()
