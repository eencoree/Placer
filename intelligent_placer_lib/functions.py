from intelligent_placer_lib import rc
import PIL
from PIL import Image
import numpy as np
import imutils
import cv2

# def checking_path(items_path: list, items_test_path: list, polygons_path: list) -> rc.RC or tuple[rc.RC, rc.RC]:
#     for it_path in items_path:
#         if not os.path.exists(it_path):
#             return rc.RC.RC_ERROR_CHECKING_PATH, rc.RC.RC_ITEMS_PATH_ERROR
#     for it_test_path in items_test_path:
#         if not os.path.exists(it_test_path):
#             return rc.RC.RC_ERROR_CHECKING_PATH, rc.RC.RC_ITEMS_TEST_PATH_ERROR
#     for pol_path in polygons_path:
#         if not os.path.exists(pol_path):
#             return rc.RC.RC_ERROR_CHECKING_PATH, rc.RC.RC_POLYGONS_PATH_ERROR
#     return rc.RC.RC_SUCCESS


def compress_img(image_name, new_size_ratio=0.9, width=None, height=None) -> Image or rc.RC:
    try:
        img = Image.open(image_name)
    except PIL.UnidentifiedImageError:
        return rc.RC.RC_OPENING_IMAGE_ERROR
    if new_size_ratio < 1.0:
        img = img.resize((int(img.size[0] * new_size_ratio), int(img.size[1] * new_size_ratio)), Image.ANTIALIAS)
    elif width and height:
        img = img.resize((width, height), Image.ANTIALIAS)
    return img


def preproc_img(filepath: str):
    # Image compression
    img = compress_img(filepath)
    if type(img) is rc.RC:
        return img
    # Converting an image to an array
    img = np.array(img)

    # Converting an image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img_gray


def cropping_img(img: np.ndarray) -> np.ndarray or tuple[rc.RC, rc.RC]:
    try:
        thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

        # Find contour and sort by contour area
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except cv2.error:
        return rc.RC.RC_CROPPING_IMG_ERROR, rc.RC.RC_CV2_ERROR

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Find bounding box and extract ROI
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cropped_img = img[y:y + h, x:x + w]
        break
    return cropped_img


def find_contour(img: np.ndarray) -> int or tuple[rc.RC, rc.RC]:
    try:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(img, 50, 200, (5, 5), L2gradient=True)
        # binarize the image
        binr = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # define the kernel
        kernel = np.ones((2, 2), np.uint8)
        closing = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        area = cv2.contourArea(contours[0])  # polygon area
    except cv2.error:
        return rc.RC.RC_FIND_CONTOUR_ERROR, rc.RC.RC_CV2_ERROR
    return area


def find_marker(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    # find the contours in the edged image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # compute the bounding box of the paper region and return it
    return cv2.minAreaRect(c)


def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth


def calc_focal_length(sheet: np.ndarray, img: np.ndarray or int):
    known_distance = 31
    # initialize the known object width, which in this case, the piece of
    known_width = 21
    # from our camera, then find the paper marker in the image, and initialize
    # the focal length

    marker = find_marker(sheet)
    focal_length = (marker[0][1] * known_distance) / known_width
    # load the image, find the marker in the image, then compute the
    # distance to the marker from the camera
    if type(img) is np.ndarray:
        marker = find_marker(img)
        inches = distance_to_camera(known_width, focal_length, marker[0][1])
    else:
        inches = distance_to_camera(known_width, focal_length, img)
    return inches
