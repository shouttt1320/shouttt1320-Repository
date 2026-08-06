# -*- coding: utf-8 -*-
"""
리센느(RESCENE) 멤버 이미지 크롤링 및 라벨링 스크립트

사용 목적:
  딥러닝 이미지 분류 모델 실습을 위한 학습 데이터셋 수집
  - Bing/Google 이미지 검색을 통한 멤버별 이미지 자동 수집
  - 디렉터리 구조 기반 자동 라벨링 (Class Label = 폴더명)
  - 손상된 이미지 필터링 및 RGB 표준 변환

멤버 목록:
  1. 원이 (WONI)
  2. 리브 (LIVV)
  3. 미나미 (MINAMI)
  4. 메이 (MAY)
  5. 제나 (ZENA)
"""

import os
import sys
import glob
from PIL import Image
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler

# UTF-8 출력 설정 (윈도우 콘솔 한글 및 이모지 깨짐 방지)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 수집할 리센느 멤버 설정 (라벨명: [검색 키워드 목록])
MEMBERS = {
    "woni": ["리센느 원이", "RESCENE Woni"],
    "livv": ["리센느 리브", "RESCENE Livv"],
    "minami": ["리센느 미나미", "RESCENE Minami"],
    "may": ["리센느 메이", "RESCENE May"],
    "zena": ["리센느 제나", "RESCENE Zena"]
}

def crawl_member_images(base_dir="dataset", max_num_per_member=50, crawler_type="google"):
    """
    멤버별 이미지를 수집하고 저장하는 함수
    
    :param base_dir: 데이터셋 저장 루트 디렉터리
    :param max_num_per_member: 멤버당 수집할 이미지 수
    :param crawler_type: 'bing' 또는 'google'
    """
    os.makedirs(base_dir, exist_ok=True)
    print("=" * 60)
    print(f"🚀 리센느 멤버 이미지 크롤링 시작 (엔진: {crawler_type.upper()}, 멤버당 {max_num_per_member}장)")
    print("=" * 60)

    for label, keywords in MEMBERS.items():
        member_dir = os.path.join(base_dir, label)
        os.makedirs(member_dir, exist_ok=True)
        
        print(f"\n📸 [{label.upper()}] 이미지 크롤링 중...")
        
        # 키워드별 분할 수집 (여러 검색어 조합)
        num_per_keyword = max(1, max_num_per_member // len(keywords))
        
        for kw in keywords:
            print(f"  🔍 키워드: '{kw}' 검색 중...")
            
            if crawler_type.lower() == "bing":
                crawler = BingImageCrawler(
                    storage={"root_dir": member_dir},
                    log_level=30  # 경고/에러만 출력 (Clean Output)
                )
            else:
                crawler = GoogleImageCrawler(
                    storage={"root_dir": member_dir},
                    log_level=30
                )
            
            crawler.crawl(keyword=kw, max_num=num_per_keyword)
            
    print("\n✅ 모든 멤버 이미지 다운로드가 완료되었습니다.")
    print("🧹 손상된 이미지 필터링 및 파일명 정리를 진행합니다...\n")

def clean_and_format_dataset(base_dir="dataset"):
    """
    다운로드된 이미지를 검증하고 손상된 이미지 삭제 및 일관된 파일명 변경
    """
    total_valid = 0
    print("=" * 60)
    print("🔍 데이터셋 검증 및 자동 라벨링 결과")
    print("=" * 60)
    
    for label in sorted(os.listdir(base_dir)):
        member_dir = os.path.join(base_dir, label)
        if not os.path.isdir(member_dir):
            continue
        
        # 지원하는 이미지 확장자
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(member_dir, ext)))
        
        valid_count = 0
        for i, filepath in enumerate(image_files, 1):
            try:
                # 이미지 손상 여부 확인 및 RGB 변환
                with Image.open(filepath) as img:
                    img.verify()  # 이미지 손상 체크
                
                # 재오픈하여 RGB 변환 후 정규화 저장 (.jpg)
                with Image.open(filepath) as img:
                    rgb_img = img.convert('RGB')
                    new_filename = f"{label}_{valid_count + 1:04d}.jpg"
                    new_filepath = os.path.join(member_dir, new_filename)
                    rgb_img.save(new_filepath, 'JPEG', quality=95)
                
                # 원래 파일과 새 파일명이 다르면 기존 파일 삭제
                if os.path.abspath(filepath) != os.path.abspath(new_filepath):
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        
                valid_count += 1
            except Exception as e:
                # 손상되거나 읽을 수 없는 파일 삭제
                if os.path.exists(filepath):
                    os.remove(filepath)

        total_valid += valid_count
        print(f"  🏷️ 클래스 라벨: '{label:<8}' | 유효한 이미지: {valid_count:>3} 장")

    print("-" * 60)
    print(f"📊 총 라벨 수: {len(MEMBERS)}개 | 총 유효 이미지 수: {total_valid} 장")
    print(f"📁 저장 경로: {os.path.abspath(base_dir)}")
    print("=" * 60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="리센느 멤버 이미지 크롤러 및 데이터셋 라벨러")
    parser.add_argument("--max_num", type=int, default=30, help="멤버당 수집할 이미지 수 (기본값: 30)")
    parser.add_argument("--output_dir", type=str, default="dataset", help="저장할 디렉터리 (기본값: dataset)")
    parser.add_argument("--engine", type=str, default="bing", choices=["bing", "google"], help="크롤링 엔진 (bing 또는 google)")
    
    args = parser.parse_args()

    # 1. 크롤링 실행
    crawl_member_images(
        base_dir=args.output_dir,
        max_num_per_member=args.max_num,
        crawler_type=args.engine
    )

    # 2. 데이터셋 정제 및 리보맷
    clean_and_format_dataset(base_dir=args.output_dir)
