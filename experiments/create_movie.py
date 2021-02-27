import sys
import cv2
# dir_name = str(input('enter directory name : '))
images_path = 'data/' + '2021-02-27-22-13/'
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(images_path+'/pose.mp4',fourcc, 20.0, (640, 480))

if not video.isOpened():
    print("can't be opened")
    sys.exit()
print('start createing movie ... ')
i = 0
while True:
    try:
        img = cv2.imread(images_path+'/images/%d.png' % i)

        if img is None:
            print("can't read")
            break
        video.write(img)
        if i%100==0:
            print('image: -- %d' % i )
        i += 1
    except:
        break

video.release()
print('written')
