
import os

# 파일이 저장된 디렉토리 경로
directory = 'C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/test/'

# 디렉토리 내 모든 파일명을 가져옴
for filename in os.listdir(directory):
    # 파일명이 jpg로 끝나는 경우에만 처리
    if filename.endswith('.jpg'):
        # 숫자와 확장자를 분리
        name_part, ext = os.path.splitext(filename)
        # 숫자 부분을 5자리로 패딩 (00001, 00010 등)
        new_name = name_part.zfill(5) + ext
        # 파일명 변경
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))