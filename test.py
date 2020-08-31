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



def add_new_id(encode_dict, name, encode_list):
    """
    encode 딕셔너리에 사용자를 추가
    :param encode_dict: 딕셔너리나 리스트는 call by reference로 호출됨
    :param name: 이름, string 자료형
    :param encode_list: 딕셔너리나 리스트는 call by reference로 호출됨
    :return: 없다
    """
    if encode_dict.get(name) is None:  # 동명이인이 없으면
        encode_dict[name] = encode_list
    else:  # 동명이인이 있음
        pass  # 미구현


def face_distance(encode, path, name):

    if encode.get(name) is None:
        return -1

    compare_encode = face_encode_path(path)
    distance = np.linalg.norm(encode[name] - compare_encode)
    print('유사도(낮을수록 정확):', end='')
    print(distance)
    return distance


def main():
    """
    ./faces 폴더에 얼굴 이미지파일이 있음
    """

    # encoding 시작
    encode = {}
    for dir_path, dir_name, file_name in os.walk("./faces"):
        for f in file_name:
            if f.endswith(".jpg") or f.endswith(".png"):
                encoding = face_encode_path("faces/" + f)
                name = f.split(".")[0]  # f.split(".")[0] 은 해당 파일의 이름.
                encode[name] = encoding  # name을 key값으로 가지는 데이터
    # encoding 끝

    # 임의 사용자 추가 시작
    add_name = '???'
    add_path = "./add_test.jpg"
    add_encode = face_encode_path(add_path)
    add_new_id(encode, add_name, add_encode)
    # 임의 사용자 추가 끝

    # face recognition 시작
    test_name = 'yuil'
    test_path = './yuil.jpg'
    distance = face_distance(encode, test_path, test_name)
    if distance is -1:
        print('존재하지않는 사용자')
    elif distance < 0.5:
        print('통과')
    else:
        print('다른사람임')
    # face recognition 끝



main()

