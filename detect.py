from aligndlib import AlignDlib

face_landmarks = 'data/shape_predictor_68_face_landmarks.dat'

def find_face(image, image_dimension):
    """
    :param images:
    :return:cropped

    """
    align_dlib = AlignDlib(face_landmarks)
    bb = align_dlib.getAllFaceBoundingBoxes(image)
    cropped = None
    for box in bb:
        cropped = align_dlib.align(image_dimension, image, box)
    return cropped, bb
