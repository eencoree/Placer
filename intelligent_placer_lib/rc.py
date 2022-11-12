import enum


class RCModule(str, enum.Enum):
    functions = "functions",


class RCType(enum.IntEnum):
    CODE_SUCCESS = 1
    CODE_FILEPATH_ERROR = 2
    CODE_NULL_ARGUMENT = 3
    INVALID_INPUT_FORMAT = 4
    CODE_NONE = 5
    CODE_ERROR_CHECKING_PATH = 6
    CODE_ERROR_OPENING_IMAGE = 7
    CODE_PREPROC_IMG_ERROR = 8
    CODE_CV2_ERROR = 9
    CODE_CROPPING_IMG_ERROR = 10
    CODE_FIND_CONTOUR_ERROR = 11


class RC:
    class ReturnCode:
        def __init__(self, Who: RCModule, Type: RCType, info: str):
            self.who = Who
            self.type = Type
            self.info = info

        def __str__(self):
            return str(self.who.value) + " " + str(self.type) + " " + str(self.info)

    @staticmethod
    def is_success(code: ReturnCode) -> bool:
        return code.type == RCType.CODE_SUCCESS

    RC_SUCCESS = ReturnCode(RCModule.functions, RCType.CODE_SUCCESS, "No errors were found")
    RC_OPENING_IMAGE_ERROR = ReturnCode(RCModule.functions, RCType.CODE_ERROR_OPENING_IMAGE, "Image cannot be opened and "
                                                                                          "identified")
    RC_PREPROC_IMG_ERROR = ReturnCode(RCModule.functions, RCType.CODE_PREPROC_IMG_ERROR, "The method preproc_img worked "
                                                                                      "with an error")
    RC_CROPPING_IMG_ERROR = ReturnCode(RCModule.functions, RCType.CODE_CROPPING_IMG_ERROR,
                                       "The method cropping_img worked "
                                       "with an error")
    RC_CV2_ERROR = ReturnCode(RCModule.functions, RCType.CODE_CV2_ERROR, "Methods cv2. worked "
                                                                      "with an error")
    RC_FIND_CONTOUR_ERROR = ReturnCode(RCModule.functions, RCType.CODE_FIND_CONTOUR_ERROR,
                                       "The method find_contour worked "
                                       "with an error")
