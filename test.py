import face_recognition
import PIL.Image
import numpy as np
import os

"""
 주의 : face_distance() 주석 볼것
"""


def face_encode_path(path, mode='RGB'):
    """
    이미지 파일을 읽어와서 데이터를 인코딩
    :param path: 이미지 파일의 주소, string
    :param mode: 컬러 이미지:'RGB' 흑백 이미지:'L'
    :return: encoding된 얼굴의 데이터, float 자료형 np.array(길이:128)
    """
    im = PIL.Image.open(path)
    if mode:
        im = im.convert(mode)
    encoding = face_recognition.face_encodings(np.array(im))[0]
    return encoding


def face_encode_img(img):
    """
    이미지 array를 받아와서 데이터를 인코딩
    :param img: 이미지 array, N * N * 3 np.array
    :return: encoding된 얼굴의 데이터, float 자료형 np.array(길이:128)
    """
    encoding = face_recognition.face_encodings(img)[0]
    return encoding


def face_distance(face_encodings, face_to_compare):
    """
    기존의 인코딩된 얼굴데이터들과 새로운 데이터를 비교한 값들을 리턴해줌
    :param faces: encoding된 데이터 리스트, List (List in N * N * 3 np.array ) )
    :param face_to_compare: 비교할 얼굴 데이터, float 자료형 np.array(길이:128)
    :return: 비교된 결과, np.array
    """
    """
    face_distance 사용법 
        face_encodings에 들어온 데이터중 가장 비슷한 얼굴의 index는 리턴된 배열에서 가장 낮은 값을 가지는 값의 index와 같다.
        0.35~0.5이하의 값을 가지면 동일인으로 볼 수 있다. (주의 : face_recognition 라이브러리의 compare_faces()은 
        서양인 기준이라서 0.6으로 되어있어서 부정확 하기 때문에 for문으로 직접 찾아 줘야한다..)
        """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

def main():
    """
    ./faces 폴더에 얼굴 이미지파일이 있음
    """

    # encoding 시작
    encode = {}  #
    for dir_path, dir_name, file_name in os.walk("./faces"):
        for f in file_name:
            if f.endswith(".jpg") or f.endswith(".png"):
                encoding = face_encode_path("faces/" + f)
                name = f.split(".")[0]  # f.split(".")[0] 은 해당 파일의 이름.
                encode[name] = encoding  # name을 key값으로 가지는 데이터
    # encoding 끝

    # face recognition 시작
    path = "./yuil.jpg"  # 찾을 이미지
    face_to_compare = face_encode_path(path)
    min_distance_name = "unknown"
    min_distance = 0.5
    for name in encode.keys():
        distance = np.linalg.norm(encode[name] - face_to_compare)
        print(name + '`s distance : ' + distance)
        if distance < min_distance:
            min_distance = distance;
            min_distance_name = name

    print('\nname : ' + min_distance_name)
    # face recognition 끝


main()

