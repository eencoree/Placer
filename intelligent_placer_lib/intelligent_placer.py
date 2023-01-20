from scipy.ndimage import binary_fill_holes
from skimage import measure
from skimage.measure import regionprops
from intelligent_placer_lib import rc
import PIL
from PIL import Image
import numpy as np
import imutils
import cv2
from time import time
from matplotlib import pyplot as plt

Angle = 5


def check_image(filepath_objects: str, filepath_polygon: str):
    flag = False
    img_pol = preproc_img(filepath_polygon)
    if type(img_pol) is rc.RC:
        print(img_pol)
        return None
    cropped_polygon = cropping_img(img_pol)
    if type(cropped_polygon) is tuple:
        print(x for x in cropped_polygon)
        return None
    _, pol_area, pol_prop = find_contour(cropped_polygon)

    img = preproc_img(filepath_objects)
    if type(img) is rc.RC:
        print(img)
        return None
    cropped_item = cropping_img(img)
    if type(cropped_item) is tuple:
        print(x for x in cropped_item)
        return None
    _, mask_area, prp = find_objects(cropped_item)
    if mask_area > pol_area:
        print("Оценочная упаковка по сумме площадей невозможна")
        with open("results.txt", 'a+') as f:
            f.write(f"файл объектов: {filepath_objects.split('/')[-1]};многоугольник: {filepath_polygon.split('/')[-1]}"
                    f";результат: false\n")
        return None
    else:
        print("Предварительно предметы можно упаковать")
        img_pol = Image.fromarray((pol_prop.image * 255).astype("uint8"))
        img_pol = np.asarray(img_pol)
        prop_polygon = measure.regionprops(measure.label(img_pol, connectivity=1))
        polygon_area = sum_area(prop_polygon)
        for prop in prop_polygon:
            if prop.area > 1000:
                prop_polygon = prop
                break
        plt.imshow(img_pol, cmap='gray')
        plt.show()
        prp.sort(key=return_area)
        prp.reverse()
        t1 = time()
        for elem in prp:
            img_obj = (elem.image * 255).astype("uint8")
            prop_img = measure.regionprops(measure.label(img_obj, connectivity=1))
            for props in prop_img:
                if props.area > 1000:
                    prop_img = props
            if prop_img.feret_diameter_max > prop_polygon.feret_diameter_max:
                print("Следующий предмет не помещается по длине")
                plt.imshow(np.asarray((prop_img.image * 255).astype("uint8")), cmap='gray')
                plt.show()
                with open("results.txt", 'a+') as f:
                    f.write(
                        f"файл объектов: {filepath_objects.split('/')[-1]};многоугольник: {filepath_polygon.split('/')[-1]}"
                        f";результат: false\n")
                flag = True
                break
            layer = Image.new("L", (np.asarray(img_pol).shape[1], np.asarray(img_pol).shape[0]), 0)
            layer.paste(Image.fromarray(img_obj))
            layer = np.asarray(layer)
            start, img_res = pos_start(layer, img_pol, prop_img, polygon_area, prop_polygon)
            prop_img = measure.regionprops(measure.label(img_res, connectivity=1))
            for props in prop_img:
                if props.area > 1000:
                    prop_img = props
            img_res = packing(start, polygon_area, img_pol, prop_img)
            if img_res is not None:
                img_pol = img_pol - img_res
                polygon_area = recalculation_area(img_pol)
                plt.imshow(img_pol, cmap='gray')
                plt.show()
            else:
                print("Не удалось уместить следующий предмет:")
                plt.imshow(np.asarray(img_obj), cmap='gray')
                plt.show()
                with open("results.txt", 'a+') as f:
                    f.write(
                        f"файл объектов: {filepath_objects.split('/')[-1]};многоугольник: {filepath_polygon.split('/')[-1]}"
                        f";результат: false\n")
                flag = True
                print("Упаковка, на которой остановились:")
                plt.imshow(img_pol, cmap='gray')
                plt.show()
                break
        if flag is False:
            with open("results.txt", 'a+') as f:
                f.write(
                    f"файл объектов: {filepath_objects.split('/')[-1]};многоугольник: {filepath_polygon.split('/')[-1]}"
                    f";результат: true\n")
        print(f'Время упаковки составило: {time() - t1} сек')


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
    img = np.array(img)

    # Converting an image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


def cropping_img(img: np.ndarray) -> np.ndarray or tuple[rc.RC, rc.RC]:
    try:
        thresh = cv2.threshold(img, 100, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        # Find contour and sort by contour area
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except cv2.error:
        return rc.RC.RC_CROPPING_IMG_ERROR, rc.RC.RC_CV2_ERROR

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Find bounding box and extract ROI
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cropped_img = img[y + 50:y + h - 50, x + 50:x + w - 50]
        break
    return cropped_img


def find_contour(img: np.ndarray):
    try:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(img, 70, 200, (3, 3), L2gradient=True)
        # binarize the image
        binr = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # define the kernel
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=2)
        plt.imshow(closing, cmap='gray')
        plt.show()
        edge_segmentation = binary_fill_holes(closing)
        edge_segmentation = (edge_segmentation * 255).astype("uint8")

        # collecting connectivity labels from the image
        labels = measure.label(edge_segmentation, connectivity=1)
        properties = regionprops(labels)

        # to determine our polygon, we will put the points of the center of mass of each found area
        center = (img.shape[0] / 2, img.shape[1] / 2)

        # After cropping the photos on the sheet, fragments of the background could remain, and they were determined as
        # areas of connectivity. Therefore, of all the areas found,
        # the one we need will be closest to the center of the sheet
        dist = np.array(
            [pow(center[0] - prop.centroid[0], 2) + pow(center[1] - prop.centroid[1], 2) for prop in properties])
        edge_segmentation_index = dist.argmin()
        edge_segmentation_mask = (labels == properties[edge_segmentation_index].label)
        plt.imshow(edge_segmentation_mask, cmap='gray')
        plt.show()
        pol_area = properties[edge_segmentation_index].area
    except cv2.error:
        return rc.RC.RC_FIND_CONTOUR_ERROR, rc.RC.RC_CV2_ERROR
    return edge_segmentation_mask, pol_area, properties[edge_segmentation_index]


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
    known_distance = 32
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
        dist = distance_to_camera(known_width, focal_length, marker[0][1])
    else:
        dist = distance_to_camera(known_width, focal_length, img)
    return dist


def find_objects(img: np.ndarray):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    my_edge_map = cv2.Canny(img_gray, 20, 100, (7, 7), L2gradient=True)
    plt.imshow(my_edge_map, cmap='gray')
    correct_mask_borders_after_canny(my_edge_map)
    plt.imshow(my_edge_map, cmap='gray')
    binr = cv2.threshold(my_edge_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((7, 7), np.uint8)
    closing = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=7)
    plt.imshow(closing, cmap='gray')
    edge_segmentation = binary_fill_holes(closing)
    edge_segmentation = (edge_segmentation * 255).astype("uint8")
    # collecting connectivity labels from the image
    labels = measure.label(edge_segmentation, connectivity=1)
    properties = regionprops(labels)
    clean_props = []

    for prop in properties:
        if 300 <= prop.centroid[0] <= (img_gray.shape[0] - 300) and 300 <= prop.centroid[1] <= (
                img_gray.shape[1] - 300) and prop.area > 40000:
            clean_props.append(prop)
    images = []
    for elem in clean_props:
        image = (labels == elem.label)
        image = (image * 255).astype("uint8")
        images.append(image)
    if len(images) == 1:
        edge_segmentation_mask = image
    else:
        edge_segmentation_mask = conc_imgs(images)
    plt.imshow(edge_segmentation_mask, cmap='gray')
    plt.show()
    return edge_segmentation_mask, sum_area(clean_props), clean_props


def conc_imgs(img_list):
    n = len(img_list)
    tmp_image = img_list[0]
    for i in range(n - 1):
        tmp_image = cv2.addWeighted(tmp_image, 1, img_list[i + 1], 1, 0.0)
    return tmp_image


def pos_start(img_object, img_polygon, prps, area_pol, polygon_prp):
    global res_img
    min_area = area_pol
    start_pos = 0
    y = polygon_prp.bbox[2]
    x = polygon_prp.bbox[3]
    for i in range(4):
        pos = (0, 0)
        copy_img_obj = img_object.copy()
        if i == 1:
            pos = (x - prps.bbox[3], 0)
        if i == 2:
            pos = (x - prps.bbox[3], y - prps.bbox[2])
        if i == 3:
            pos = (0, y - prps.bbox[2])

        if i != 0:
            copy_img_obj = (prps.image * 255).astype("uint8")
            lr = Image.new("L", (np.asarray(img_polygon).shape[1], np.asarray(img_polygon).shape[0]), 0)
            lr.paste(Image.fromarray(copy_img_obj), pos)
            copy_img_obj = np.asarray(lr)
        res = img_polygon - copy_img_obj
        prop_res = measure.regionprops(measure.label(res, connectivity=1))
        res_area = sum_area(prop_res)
        if res_area < min_area:
            min_area = res_area
            start_pos = pos
            res_img = copy_img_obj
    if min_area == area_pol:
        lr = Image.new("L", (np.asarray(img_polygon).shape[1], np.asarray(img_polygon).shape[0]), 0)
        start_pos = (0, 0)
        lr.paste(Image.fromarray((prps.image * 255).astype("uint8")), start_pos)
        res_img = np.asarray(lr)
    return start_pos, res_img


def packing(position, polygon_area, img_polygon, prop_img):
    eccentric = prop_img.eccentricity
    global tmp_img
    step_y = round((prop_img.bbox[2] - prop_img.bbox[0]) / 6)
    step_x = round((prop_img.bbox[3] - prop_img.bbox[1]) / 6)
    a_min = 6 * min(step_x, step_y)
    tmp_img = (prop_img.image * 255).astype("uint8")
    copy_img = tmp_img.copy()
    ar = recalculation_area(tmp_img)
    if position == (0, 0):
        for y in range(0, img_polygon.shape[0] - a_min, round(step_y)):
            for x in range(0, img_polygon.shape[1] - a_min, round(step_x)):
                angle = 0
                while angle != 180:
                    lr = Image.new("L", (img_polygon.shape[1], img_polygon.shape[0]), 0)
                    lr.paste(Image.fromarray(tmp_img), (x, y))
                    tmp_img = np.asarray(lr)
                    res = img_polygon - tmp_img
                    prop_res = measure.regionprops(measure.label(res, connectivity=1))
                    res_area = sum_area(prop_res)
                    area_pack = abs(res_area - (polygon_area - ar))
                    if area_pack < 150:
                        return tmp_img
                    if eccentric < 0.4:
                        break
                    angle += Angle
                    tmp_img = np.asarray(Image.fromarray(copy_img).rotate(angle, expand=True))
                    prp_img = measure.regionprops(measure.label(tmp_img, connectivity=1))
                    for p in prp_img:
                        if p.area > 1000:
                            prp_img = p
                            break
                    tmp_img = (prp_img.image * 255).astype("uint8")
                    ar = recalculation_area(tmp_img)
                tmp_img = copy_img
                ar = recalculation_area(tmp_img)
        return None

    if position[1] == 0:
        list_x = list(range(0, img_polygon.shape[1] - a_min, round(step_x)))
        list.reverse(list_x)
        for y in range(0, img_polygon.shape[0] - a_min, round(step_y)):
            for x in list_x:
                angle = 0
                while angle != 180:
                    lr = Image.new("L", (img_polygon.shape[1], img_polygon.shape[0]), 0)
                    lr.paste(Image.fromarray(tmp_img), (x, y))
                    tmp_img = np.asarray(lr)
                    res = img_polygon - tmp_img
                    prop_res = measure.regionprops(measure.label(res, connectivity=1))
                    res_area = sum_area(prop_res)
                    area_pack = abs(res_area - (polygon_area - ar))
                    if area_pack < 150:
                        return tmp_img
                    if eccentric < 0.4:
                        break
                    angle += Angle
                    tmp_img = np.asarray(Image.fromarray(copy_img).rotate(angle, expand=True))
                    prp_img = measure.regionprops(measure.label(tmp_img, connectivity=1))
                    for p in prp_img:
                        if p.area > 1000:
                            prp_img = p
                            break
                    tmp_img = (prp_img.image * 255).astype("uint8")
                    ar = recalculation_area(tmp_img)
                tmp_img = copy_img
                ar = recalculation_area(tmp_img)
        return None

    if position[0] != 0 and position[1] != 0:
        list_y = list(range(0, img_polygon.shape[0] - a_min, round(step_y)))
        list.reverse(list_y)
        list_x = list(range(0, img_polygon.shape[1] - a_min, round(step_x)))
        list.reverse(list_x)
        for y in list_y:
            for x in list_x:
                angle = 0
                while angle != 180:
                    lr = Image.new("L", (img_polygon.shape[1], img_polygon.shape[0]), 0)
                    lr.paste(Image.fromarray(tmp_img), (x, y))
                    tmp_img = np.asarray(lr)
                    res = img_polygon - tmp_img
                    prop_res = measure.regionprops(measure.label(res, connectivity=1))
                    res_area = sum_area(prop_res)
                    area_pack = abs(res_area - (polygon_area - ar))
                    if area_pack < 150:
                        return tmp_img
                    if eccentric < 0.4:
                        break
                    angle += Angle
                    tmp_img = np.asarray(Image.fromarray(copy_img).rotate(angle, expand=True))
                    prp_img = measure.regionprops(measure.label(tmp_img, connectivity=1))
                    for p in prp_img:
                        if p.area > 1000:
                            prp_img = p
                            break
                    tmp_img = (prp_img.image * 255).astype("uint8")
                    ar = recalculation_area(tmp_img)
                tmp_img = copy_img
                ar = recalculation_area(tmp_img)
        return None

    else:
        list_y = list(range(0, img_polygon.shape[0] - a_min, round(step_y)))
        list.reverse(list_y)
        for y in list_y:
            for x in range(0, img_polygon.shape[1] - a_min, round(step_x)):
                angle = 0
                while angle != 180:
                    lr = Image.new("L", (img_polygon.shape[1], img_polygon.shape[0]), 0)
                    lr.paste(Image.fromarray(tmp_img), (x, y))
                    tmp_img = np.asarray(lr)
                    res = img_polygon - tmp_img
                    prop_res = measure.regionprops(measure.label(res, connectivity=1))
                    res_area = sum_area(prop_res)
                    area_pack = abs(res_area - (polygon_area - ar))
                    if area_pack < 150:
                        return tmp_img
                    if eccentric < 0.4:
                        break
                    angle += Angle
                    tmp_img = np.asarray(Image.fromarray(copy_img).rotate(angle, expand=True))
                    prp_img = measure.regionprops(measure.label(tmp_img, connectivity=1))
                    for p in prp_img:
                        if p.area > 1000:
                            prp_img = p
                            break
                    tmp_img = (prp_img.image * 255).astype("uint8")
                    ar = recalculation_area(tmp_img)
                tmp_img = copy_img
                ar = recalculation_area(tmp_img)
        return None
    return None


def recalculation_area(obj_image):
    obj_prop = measure.regionprops(measure.label(obj_image, connectivity=1))
    obj_area = sum_area(obj_prop)
    return obj_area


def return_area(prop):
    return prop.area


def sum_area(reg):
    sum = 0
    for elem in reg:
        sum += elem.area
    return sum


def correct_mask_borders_after_canny(canny_result, border_width=3):
    canny_result[:border_width, :] = 0
    canny_result[:, :border_width] = 0
    canny_result[-border_width:, :] = 0
    canny_result[:, -border_width:] = 0
