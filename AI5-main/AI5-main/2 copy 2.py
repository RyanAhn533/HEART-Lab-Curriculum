import os
import subprocess

# 깃허브 저장소 URL
remote_repo_url = "https://github.com/RyanAhn533/AI5.git"
# 저장소 디렉토리 설정
repo_path = "./AI5"


# 저장소 내 파일 목록 확인
for file_name in os.listdir(repo_path):
    # 'file_YYYY-MM-DD.txt' 형태의 파일만 삭제
    if file_name.startswith("file_") and file_name.endswith(".txt"):
        file_path = os.path.join(repo_path, file_name)
        os.remove(file_path)  # 파일 삭제

# Git에 변경 사항 반영
subprocess.run(["git", "add", "."], cwd=repo_path)
subprocess.run(["git", "commit", "-m", "Delete auto-generated files"], cwd=repo_path)

# 변경 사항 푸시
subprocess.run(["git", "push", "origin", "main"], cwd=repo_path)

print("파일 삭제 및 커밋 완료!")