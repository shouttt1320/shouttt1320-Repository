import os
import cv2
import subprocess
import sys

def download_youtube_video(youtube_url, output_path="temp_video.mp4"):
    """yt-dlp를 이용하여 ffmpeg 없이 단일 MP4 영상 파일로 다운로드"""
    print(f"\n[1/3] 유튜브 영상 다운로드 중... ({youtube_url})")

    # 기존 임시 파일 제거
    for f in [output_path, "temp_video.f401.mp4", "temp_video.f140.m4a"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    # ffmpeg 설치 없이 바로 읽을 수 있는 단일 mp4 포맷 및 403 Forbidden 우회 클라이언트 설정
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--extractor-args", "youtube:player_client=android,web",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "-f", "best[ext=mp4]/b[ext=mp4]/best",
        "--no-playlist",
        "-o", output_path,
        youtube_url
    ]


    try:
        subprocess.run(cmd, check=True)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"], check=True)
        subprocess.run(cmd, check=True)

    # 파일 존재 확인 및 대체파일명 탐색
    if os.path.exists(output_path):
        print(f"다운로드 완료: {output_path}")
        return output_path

    # ffmpeg 미설치로 temp_video.fxxx.mp4 형태로 다운로드 되었을 경우
    for file in os.listdir("."):
        if file.startswith("temp_video") and file.endswith(".mp4"):
            print(f"다운로드 완료 (대체 탐색): {file}")
            return file

    return output_path


def extract_faces_from_video(video_path, person_name="woni", max_count=200, sample_interval=5):
    """영상에서 얼굴을 탐지하여 224x224 크기로 크롭하여 이어서 저장"""
    save_dir = os.path.join("./face_dataset", person_name)
    os.makedirs(save_dir, exist_ok=True)

    # 기존에 저장된 파일 번호 탐색 (덮어쓰지 않고 이어서 저장)
    existing_files = [f for f in os.listdir(save_dir) if f.startswith(f"{person_name}_") and f.endswith(".jpg")]
    existing_indices = []
    for f in existing_files:
        try:
            num_part = f.replace(f"{person_name}_", "").replace(".jpg", "")
            existing_indices.append(int(num_part))
        except ValueError:
            pass

    start_index = max(existing_indices) + 1 if existing_indices else 1
    print(f"\n[2/3] 얼굴 탐지 및 데이터 수집 시작...")
    print(f" - 저장 위치: {save_dir}")
    print(f" - 시작 번호: {person_name}_{start_index:04d}.jpg부터 이어서 저장")
    print(f" - 이번 영상 수집 목표: 최대 {max_count}장 (간격: {sample_interval}프레임 마다 1장)")

    # OpenCV Haar Cascade 얼굴 탐지기 로드
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"에러: 비디오 파일 '{video_path}'을 열 수 없습니다.")
        return

    current_index = start_index - 1
    new_saved_count = 0
    frame_idx = 0

    while cap.isOpened() and new_saved_count < max_count:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_interval != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            # 여유 공간(Padding)을 포함하여 크롭
            padding = int(w * 0.15)
            y1 = max(0, y - padding)
            y2 = min(frame.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(frame.shape[1], x + w + padding)

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            # 224x224 규격 크기로 리사이즈
            resized_face = cv2.resize(face_roi, (224, 224))

            current_index += 1
            new_saved_count += 1
            file_name = f"{person_name}_{current_index:04d}.jpg"
            file_path = os.path.join(save_dir, file_name)
            cv2.imwrite(file_path, resized_face)

            print(f"\r새로 수집 중: +{new_saved_count}장 -> {file_name} (총 {current_index}장)", end="")

            if new_saved_count >= max_count:
                break

    cap.release()
    print(f"\n\n[3/3] 수집 완료! 이번 영상에서 +{new_saved_count}장이 새로 추가되어 총 {current_index}장이 되었습니다.")


    # 임시 영상 삭제
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
            print(f"임시 영상 파일 삭제 완료: {video_path}")
        except Exception:
            pass

if __name__ == "__main__":
    print("=" * 60)
    print("      [1단계] 유튜브 영상 기반 인물 얼굴 일괄 수집기")
    print("=" * 60)

    person_name = input("\n1. 인물 이름을 입력하세요 (기본값: woni): ").strip()
    if not person_name:
        person_name = "woni"

    print(f"\n2. '{person_name}'의 유튜브 URL들을 입력하세요.")
    print("   - 여러 개일 경우 쉼표(,) 또는 공백으로 구분하여 입력해주세요.")
    print("   - (엔터를 누르면 예시 숏츠 URL 사용)")

    url_input = input("\n유튜브 URL(들) 입력: ").strip()

    if not url_input:
        urls = ["https://youtube.com/shorts/ILU9HNNt1bk?si=Cl8TKCLprcRxOyKA"]
    else:
        # 쉼표(,)나 줄바꿈, 공백으로 구분하여 URL 리스트 생성
        urls = [u.strip() for u in url_input.replace(",", " ").split() if u.strip()]

    print(f"\n총 {len(urls)}개의 유튜브 영상에서 '{person_name}' 얼굴 수집을 순차적으로 진행합니다!")

    for idx, url in enumerate(urls, 1):
        print(f"\n==================================================")
        print(f"  [{idx}/{len(urls)}] 번째 영상 다운로드 및 얼굴 추출 중...")
        print(f"  URL: {url}")
        print(f"==================================================")

        try:
            video_file = download_youtube_video(url, output_path=f"temp_video_{idx}.mp4")
            extract_faces_from_video(video_file, person_name=person_name, max_count=200, sample_interval=5)
        except Exception as e:
            print(f"\n에러 발생 (URL: {url}): {e}")
            print("다음 영상으로 이동합니다.")
            continue

    print(f"\n🎉 전체 {len(urls)}개 영상에 대한 '{person_name}' 얼굴 데이터 수집이 모두 완료되었습니다!")

