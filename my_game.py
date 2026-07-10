# main.py
from objects import GameManager

if __name__ == "__main__":
    # 1. 시뮬레이터 객체 생성
    simulator = GameManager()
    
    # 2. 실행
    simulator.run()