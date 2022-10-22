import cv2

from rc import RC
import functions

# Проверка на валидность путей
# items = "C:/Users/reuto/Desktop/source items/6.jpg"  # пример пути
# polygon = "C:/Users/reuto/Desktop/input data/pentagon.jpg"  # пример пути
# RC_PATH = functions.checking_path(items, polygon)
# if type(RC_PATH) is tuple:
#     print(x for x in RC_PATH)
# else:
#     preproc_rc = functions.preproc_img(items)
#     if type(preproc_rc) is tuple:
#         print(x for x in preproc_rc)
#     else:
#         img_items = functions.get_buffer()
#     cropping_rc = functions.cropping_img(img_items)
#     if type(cropping_rc) is tuple:
#         print(x for x in cropping_rc)
#     else:
#         cropped_img_items = functions.get_buffer()
#
#     preproc_rc = functions.preproc_img(polygon)
#     if type(preproc_rc) is tuple:
#         print(x for x in preproc_rc)
#     else:
#         img_polygon = functions.get_buffer()
#     cropping_rc = functions.cropping_img(img_polygon)
#     if type(cropping_rc) is tuple:
#         print(x for x in cropping_rc)
#     else:
#         cropped_img_polygon = functions.get_buffer()
#
#     # cv2.namedWindow('ROI', cv2.WINDOW_KEEPRATIO)
#     # cv2.resizeWindow('ROI', 1280, 720)
#     # cv2.imshow('ROI',cropped_img_polygon)
#     # cv2.waitKey()
#
#     find_contour_rc = functions.find_contour(cropped_img_polygon)
#     if type(find_contour_rc) is tuple:
#         print(x for x in find_contour_rc)
#     else:
#         print(functions.get_buffer())

img = cv2.imread("C:/Users/reuto/Desktop/back.jpg")
length = functions.calc_focal_length(img, "C:/Users/reuto/Desktop/pb.jpg")  # получаем расстояние от камеры до листа
