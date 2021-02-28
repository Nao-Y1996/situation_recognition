import sys
import cv2
import glob


def create(images_path, movie_path):
    images_path = images_path
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(movie_path, fourcc, 20.0, (640, 480))

    if not video.isOpened():
        print("can't be opened")
        sys.exit()
    i = 0
    while True:
        try:
            image_path = glob.glob(images_path+"/"+str(i)+".png")[0]
            # print(image_path)
            img = cv2.imread(image_path)
            cv2.putText(img,"frame:"+str(i),(400, 440),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
            # print(images_path+"/"+str(i)+"*")

            if img is None:
                print("can't read")
                break
            video.write(img)
            if i % 500 == 0:
                print('image: -- %d' % i)
            i += 1
        except:
            break
    video.release()
