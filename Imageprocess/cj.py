import os
import cv2


def clip_image(input_dir, floder, output_dir):
    images = os.listdir(input_dir + floder)

    for imagename in images:
        imagepath = os.path.join(input_dir + floder, imagename)
        img = cv2.imread(imagepath)

        path = "F:\Py3.8.2\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"

        hc = cv2.CascadeClassifier(path)

        faces = hc.detectMultiScale(img)
        i = 1
        image_save_name = output_dir + floder + imagename
        for face in faces:
            imgROI = img[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
            imgROI = cv2.resize(imgROI, (48, 48), interpolation=cv2.INTER_AREA)
            cv2.imwrite(image_save_name, imgROI)
            i = i + 1
        print("the {}th image has been processed".format(i))


def main():
    input_dir = "data/1/"
    floder = ""
    output_dir = "data/2/"

    if not os.path.exists(output_dir + floder):
        os.makedirs(output_dir + floder)

    clip_image(input_dir, floder, output_dir)

if __name__ == '__main__':
    main()