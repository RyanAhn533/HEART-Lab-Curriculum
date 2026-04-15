#이미지 넘버링
import os
#기존 파일 경로
folder_path = 'C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/test/'

#새로 저장 경로 (정확히 맞아야 함)
dst_path = 'C:/프로그램/ai5/_data/kaggle/dogs_vs_cats/test/'

#폴더안에있는 파일들 이름 리스트로 가져오기
file_names = os.listdir(folder_path)

numbering = 1

for file in file_names:
    f_name, f_format = file.split(sep = '.') #파일 이름, 파일 확장자 split

    if f_format == 'jpg': #사진이면
        if f_name + '.txt' in file_names: #라벨링이 된 사진이면
            #print(f_name, f_format)
            src_jpg = os.path.join(folder_path, f_name + '.jpg')
         

            rename_jpg = '0000_' + str(numbering) + '.jpg' # (팀원순서)_(영상순서)_(이미지 순서) + .jpg
           

            numbering += 1 #넘버링 다음
            print(rename_jpg)

            dst_jpg = os.path.join(dst_path, rename_jpg)
          

            os.rename(src_jpg, dst_jpg)
         