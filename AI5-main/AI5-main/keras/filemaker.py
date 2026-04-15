import os

# 텍스트 파일들이 있는 디렉터리 경로
directory_path = 'C:\\Users\\Ryan\\Desktop\\수업정리\\filemaker.txt'

# 디렉터리 내 모든 파일을 확인
for filename in os.listdir(directory_path):
    # 텍스트 파일만 처리
    if filename.endswith('.txt'):
        # 파일명에서 확장자(.txt)를 제거하여 새로운 파일명으로 사용
        new_filename = os.path.splitext(filename)[0] + '.py'
        
        # py 파일 생성
        with open(os.path.join(directory_path, new_filename), 'w') as new_file:
            new_file.write("# This is an auto-generated .py file.\n")
        
        print(f"{new_filename} 파일이 생성되었습니다.")

print("모든 텍스트 파일이 파이썬 파일로 변환되었습니다.")
