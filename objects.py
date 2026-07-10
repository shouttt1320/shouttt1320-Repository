# objects.py
import pygame
import random
import math

# 색상 정의
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
CYAN = (0, 255, 255)

class Car:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.width = 40
        self.height = 60
        # 화면 하단 중앙 배치
        self.x = (screen_width // 2) - (self.width // 2)
        self.y = screen_height - 100
        self.speed = 5
        self.color = GREEN

    def update(self):
        # WASD 키 조종
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]: # 왼쪽
            self.x -= self.speed
        if keys[pygame.K_d]: # 오른쪽
            self.x += self.speed
        if keys[pygame.K_w]: # 위
            self.y -= self.speed
        if keys[pygame.K_s]: # 아래
            self.y += self.speed

        # 화면 밖 탈출 방지 경계 제한
        if self.x < 0: self.x = 0
        if self.x > self.screen_width - self.width: self.x = self.screen_width - self.width
        if self.y < 0: self.y = 0
        if self.y > self.screen_height - self.height: self.y = self.screen_height - self.height

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (int(self.x), int(self.y), self.width, self.height))

    def get_rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.width, self.height)


class Obstacle:
    def __init__(self, screen_width, screen_height, base_speed, current_stage):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 25
        # 화면 위쪽 랜덤한 위치에서 생성
        self.x = random.randrange(0, screen_width - self.size)
        self.y = random.randrange(-400, -50)
        # 스테이지가 오를수록 속도의 최솟값/최댓값이 함께 증가
        self.speed = random.uniform(base_speed, base_speed + 3)
        self.color = RED
        self.current_stage = current_stage
        self.movement_type = "NORMAL"  # 기본은 직진형
        self.speed_x = 0
        
        # 2스테이지 이상부터 좌우 움직임 활성화 확률 부여
        if self.current_stage >= 2:
            # 스테이지가 높을수록 특수 패턴 장애물이 나올 확률 증가
            # (2스테이지: 40%, 3스테이지: 60%, 4스테이지 이상: 80%)
            special_chance = min(20 + (self.current_stage * 10), 80)
            
            if random.randrange(0, 100) < special_chance:
                # 패턴을 두 가지 종류로 다양화합니다.
                self.movement_type = random.choice(["WAVE", "ZIGZAG"])
                
                if self.movement_type == "WAVE":
                    self.color = (255, 0, 255)  # 🟣 웨이브 패턴은 보라색
                    self.wave_speed = random.uniform(0.02, 0.05) # 물결치는 속도
                    self.wave_amp = random.uniform(3, 7)         # 좌우 흔들림 폭
                    self.time_counter = random.uniform(0, 100)   # 시작 지점 분산용
                    
                elif self.movement_type == "ZIGZAG":
                    self.color = (255, 165, 0)  # 🟠 지그재그 패턴은 주황색
                    # 스테이지가 높을수록 좌우 튕기는 속도 증가
                    self.speed_x = random.choice([-1, 1]) * random.uniform(1.5, 1.5 + (current_stage * 0.5))

    def update(self, current_stage):
            self.current_stage = current_stage
            
            # 1. 공통적으로 아래로는 계속 떨어집니다. (speed_y 오타 수정 -> speed)
            self.y += self.speed
            
            # 2. 타입별 좌우 움직임 추가
            if self.movement_type == "WAVE":
                # 사인(sin) 함수를 이용해 부드럽게 춤추듯 좌우로 움직임
                self.time_counter += 1
                self.x += math.sin(self.time_counter * self.wave_speed) * self.wave_amp
                
            elif self.movement_type == "ZIGZAG":
                # 일정한 속도로 가다가 벽에 부딪히면 튕김
                self.x += self.speed_x
                if self.x <= 0 or self.x >= self.screen_width - self.size:
                    self.speed_x *= -1  # 방향 뒤집기

            # 🔍 디버깅용 print문
            if self.movement_type != "NORMAL":
                print(f"패턴: {self.movement_type} | 현재 X: {int(self.x)}")

            # 3. 화면 아래로 완전히 지나치면 위에서 재지정되어 리스폰
            if self.y > self.screen_height:
                # 게임 매니저에서 줬던 base_speed 공식을 다시 적용해 객체 완전 초기화 (새 패턴 뽑기)
                base_speed = 2 + (self.current_stage * 1)
                self.__init__(self.screen_width, self.screen_height, base_speed, self.current_stage)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (int(self.x), int(self.y), self.size, self.size))

    def get_rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.size, self.size)


class GameManager:
    def __init__(self):
        pygame.init()
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Car Simulator")
        self.clock = pygame.time.Clock()
        
        # 게임 시스템 변수
        self.stage = 1
        self.score = 0
        self.game_over = False
        self.running = True
        self.font = pygame.font.SysFont("arial", 30, bold=True)
        
        # 객체 생성
        self.car = Car(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.obstacles = []
        self.spawn_obstacles()

    def spawn_obstacles(self):
        # 스테이지가 올라갈수록 장애물 수 증가 (기본 5개 + 스테이지당 5개씩 추가)
        num_obstacles = 5 + (self.stage - 1) * 5
        # 스테이지가 올라갈수록 속도 증가
        base_speed = 2 + (self.stage * 1)
        
        self.obstacles = [Obstacle(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, base_speed, self.stage) for _ in range(num_obstacles)]

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        if not self.game_over:
            self.car.update()
            
            for obs in self.obstacles:
                obs.update(self.stage)  # 💡 self.stage 전달!
                
                if self.car.get_rect().colliderect(obs.get_rect()):
                    self.game_over = True
            
            # 시간의 흐름에 따라 점수 증가
            self.score += 1
            
            # 점수가 일정 기준을 넘으면 스테이지 업! (예: 500점 단위)
            if self.score >= self.stage * 1000:
                self.stage += 1
                self.spawn_obstacles() # 다음 스테이지 환경으로 장애물 리셋

    def draw(self):
        self.screen.fill(BLACK)
        
        # 객체들 그리기
        self.car.draw(self.screen)
        for obs in self.obstacles:
            obs.draw(self.screen)
            
        # UI 레이블 표시 (스테이지, 점수)
        stage_txt = self.font.render(f"STAGE: {self.stage}", True, CYAN)
        score_txt = self.font.render(f"SCORE: {self.score}", True, WHITE)
        self.screen.blit(stage_txt, (20, 20))
        self.screen.blit(score_txt, (20, 55))
        
        # 게임오버 연출
        if self.game_over:
            over_txt = self.font.render("GAME OVER", True, RED)
            over_rect = over_txt.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(over_txt, over_rect)

        pygame.display.flip()

    def run(self):
        # 메인 게임 루프
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60) # 60 FPS 고정으로 기기별 일정한 속도 보장
            
        pygame.quit()