import cv2
import util
import uuid
import os
import argparse
import numpy as np


class YATR():
    """
    Class to help detect and recognize text in images.
    """

    PL_MODEL_CONV_TYPE = "pixel_link/conv3_3"
    PL_MODEL_CHECKPOINT = "model.ckpt-38055"
    AON_MODEL_CHECKPOINT = "AON/exp_log"

    @staticmethod
    def __generate_filename():
        file_name = uuid.uuid4()
        return "/tmp/{}.jpg".format(file_name)

    @classmethod
    def crop_image(cls, image_path, coordinate_file):
        img = cv2.imread(image_path, 0)  # Read in your image
        with open(coordinate_file, 'r') as coordinate_file:
            lines = coordinate_file.readlines()
            output = []
            for line in lines:
                coordinates = list(map(lambda x: int(x), line.split(',')))
                points = np.reshape(coordinates, [4, 2])
                contours = util.img.points_to_contours(points)
                # Create mask where white is what we want, black otherwise
                mask = np.zeros_like(img)
                # Draw filled contour in mask
                cv2.drawContours(mask, contours, 0, 255, -1)
                # Extract out the object and place into output image
                out = np.zeros_like(img)
                out[mask == 255] = img[mask == 255]

                # Now crop
                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))
                out = out[topx:bottomx + 1, topy:bottomy + 1]
                out_filename = cls.__generate_filename()
                output.append(out_filename)
                cv2.imwrite(out_filename, out)
        return output

    @classmethod
    def detect_text(cls, dataset_dir):
        """
        Function to detect the text in images using Pixel Link.
        """
        cmd = "CUDA_VISIBLE_DEVICES=0 python pixel_link/test_pixel_link.py \
            --checkpoint_path={0} \
            --dataset_dir={1} \
            --gpu_memory_fraction=-1".format(
            "{}/{}".format(cls.PL_MODEL_CONV_TYPE, cls.PL_MODEL_CHECKPOINT),
            dataset_dir
        )
        os.system(cmd)
        result = []
        for file in os.listdir(dataset_dir):
            txt_file = "{}.txt".format(file.split('.')[0])
            result_filename = "{}/test/icdar2015_test/{}/txt/res_{}".format(
                cls.PL_MODEL_CONV_TYPE,
                cls.PL_MODEL_CHECKPOINT,
                txt_file
            )
            result.append([
                "{}/{}".format(dataset_dir, file),
                result_filename
            ]
            )
        return result

    @classmethod
    def recognize_text(cls, image_path):
        """
        Function to detect text using the Pixel Link Model
        """
        cmd = "python AON/test.py --mode single --image_path {} --exp_dir {}".format(
            image_path,
            cls.AON_MODEL_CHECKPOINT
        )
        os.system(cmd)

    @classmethod
    def detect(cls, dataset_dir):
        result = cls.detect_text(dataset_dir)
        for record in result:
            [image_path, coordinate_file] = record
            output = cls.crop_image(image_path, coordinate_file)
            for filename in output:
                cls.recognize_text(filename)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="Directory containing the images")
    args = parser.parse_args()
    YATR.detect(args.dataset_dir)
