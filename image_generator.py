import cv2

from detect import find_face

image_path = 'data/images'
label = ''
idx = 0
video_camera = cv2.VideoCapture(0)
while video_camera.isOpened():
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break
    ret, frame = video_camera.read()
    cropped, bb = find_face(frame, 299)
    if bb:
        cv2.imwrite(image_path + '/' + label + '/' + label + '_' + str(idx) + '.jpg', cropped)
        idx += 1
        print idx
    cv2.imshow('Camera', frame)

