import os.path
from rc import RC
import PIL
from PIL import Image
from imutils import paths
import numpy as np
import imutils
import cv2

container: object


def fill_buffer(data: object) -> RC:
    global container
    container = data
    return RC.RC_SUCCESS


def get_buffer():
    return container


def checking_path(items_path: str, polygon_path: str) -> RC or tuple(RC, RC):
    if not os.path.exists(items_path):
        return RC.RC_ERROR_CHECKING_PATH, RC.RC_ITEMS_PATH_ERROR
    elif not os.path.exists(polygon_path):
        return RC.RC_ERROR_CHECKING_PATH, RC.RC_POLYGON_PATH_ERROR
    else:
        return RC.RC_SUCCESS


def compress_img(image_name, new_size_ratio=0.9, width=None, height=None) -> RC:
    try:
        img = Image.open(image_name)
    except PIL.UnidentifiedImageError:
        return RC.RC_OPENING_IMAGE_ERROR
    if new_size_ratio < 1.0:
        img = img.resize((int(img.size[0] * new_size_ratio), int(img.size[1] * new_size_ratio)), Image.ANTIALIAS)
    elif width and height:
        img = img.resize((width, height), Image.ANTIALIAS)
    fill_buffer(img)
    return RC.RC_SUCCESS


def preproc_img(filepath: str) -> RC or tuple(RC, RC):
    # Сжатие изображения
    img_rc = compress_img(filepath)
    if RC.is_success(img_rc):
        img = get_buffer()
    else:
        return RC.RC_PREPROC_IMG_ERROR, img_rc

    # Преобразование изображения в массив
    img = np.array(img)

    # Преобразование изображения в оттенки серого
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Применяем размытие по Гауссу
    # img = cv2.GaussianBlur(img_gray, (3, 3), 0) #решил убрать так как хуже границы определяются
    fill_buffer(img_gray)
    return RC.RC_SUCCESS


def cropping_img(img: np.ndarray) -> RC or tuple(RC, RC):
    try:
        thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

        # Find contour and sort by contour area
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except cv2.error:
        return RC.RC_CROPPING_IMG_ERROR, RC.RC_CV2_ERROR

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Find bounding box and extract ROI
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cropped_img = img[y:y + h, x:x + w]
        break
    fill_buffer(cropped_img)
    return RC.RC_SUCCESS


def find_contour(img: np.ndarray) -> RC or tuple(RC, RC):
    try:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(img, 50, 200, (5, 5), L2gradient=True)
        # binarize the image
        binr = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # define the kernel
        kernel = np.ones((2, 2), np.uint8)
        closing = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        area = cv2.contourArea(contours[0])  # площадь многоугольника
        fill_buffer(area)
        cv2.namedWindow('ROI', cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('ROI', 1280, 720)
        cv2.imshow('ROI', closing)
        cv2.waitKey()
    except cv2.error:
        return RC.RC_FIND_CONTOUR_ERROR, RC.RC_CV2_ERROR
    return RC.RC_SUCCESS


def find_marker(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # compute the bounding box of the of the paper region and return it
    return cv2.minAreaRect(c)


def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth


def calc_focal_length(image: np.ndarray, filepath: str):
    known_distance = 31
    # initialize the known object width, which in this case, the piece of
    known_width = 21
    # from our camera, then find the paper marker in the image, and initialize
    # the focal length

    marker = find_marker(image)
    focal_length = (marker[0][1] * known_distance) / known_width
    # for imagePath in sorted(paths.list_images("images")):
    # load the image, find the marker in the image, then compute the
    # distance to the marker from the camera
    image = cv2.imread(filepath)
    marker = find_marker(image)
    inches = distance_to_camera(known_width, focal_length, marker[0][1])
    return inches
