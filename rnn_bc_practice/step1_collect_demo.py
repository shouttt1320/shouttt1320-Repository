import numpy as np
import mujoco
import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def collect_expert_demonstrations(xml_path="scene.xml", num_episodes=100, episode_len=100):
    print("=" * 60)
    print("🚀 [Step 1] 특정 목표 위치 도달(Point-to-Point) 전문가 궤적 수집")
    print("   (부드러운 S-커브 보간 + 다양한 궤적 노이즈 적용)")
    print("=" * 60)
    
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
        
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 🎯 1. 기준 시작 위치(Start Pose) 및 특정 도달 목표 위치(Target Goal Pose)
    # [Shoulder Pan, Shoulder Lift, Elbow Flex, Wrist Flex, Wrist Roll, Gripper]
    base_start_pose  = np.array([ 0.00, -0.40,  0.60,  0.30,  0.00,  0.10], dtype=np.float32) # 홈/대기 자세
    base_target_pose = np.array([ 0.45,  0.25,  0.25, -0.20,  0.60,  0.80], dtype=np.float32) # 특정 목표 위치 (예: 물체 집기 위치)
    
    all_states = np.zeros((num_episodes, episode_len, 6), dtype=np.float32)
    all_actions = np.zeros((num_episodes, episode_len, 6), dtype=np.float32)
    
    np.random.seed(42)  # 재현성 보장
    
    for ep in range(num_episodes):
        # 🎲 2. 에피소드마다 시작 위치와 목표 위치에 약간의 랜덤 노이즈 부여 (다양성 확보)
        start_noise  = np.random.uniform(-0.08, 0.08, size=6)
        target_noise = np.random.uniform(-0.06, 0.06, size=6)
        
        ep_start_pose  = base_start_pose + start_noise
        ep_target_pose = base_target_pose + target_noise
        
        # 🎲 3. 중간 궤적에 부드러운 왜곡(Warping Noise)을 주기 위한 무작위 주파수/위상
        mid_noise_amp  = np.random.uniform(0.02, 0.05, size=6)
        mid_noise_freq = np.random.uniform(1.0, 2.0, size=6)
        mid_noise_phase= np.random.uniform(0, np.pi, size=6)
        
        # 시뮬레이터 리셋 및 시작 자세 초기화
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = ep_start_pose
        data.ctrl[:6] = ep_start_pose
        mujoco.mj_step(model, data)
        
        for t in range(episode_len):
            # 진행률 alpha (0.0 -> 1.0): 코사인 S-커브 (Minimum-Jerk 부드러운 가속/감속)
            # 앞선 75스텝 동안 목표 위치로 이동하고, 마지막 25스텝은 목표 위치에 안정적으로 도달하여 유지
            move_steps = 75
            progress = min(1.0, t / move_steps)
            alpha = 0.5 * (1.0 - np.cos(np.pi * progress)) # 0.0에서 시작해 1.0으로 부드럽게 증가
            
            # 기본 S-커브 보간 궤적
            interpolated_pose = (1.0 - alpha) * ep_start_pose + alpha * ep_target_pose
            
            # 이동 중에만 살짝 나타났다가 목표 지점에 도착하면 0이 되는 중간 경로 노이즈 (Sinusoidal Envelope)
            # t=0일 때 0, t=75일 때 0, 중간(t=37)에서 최대 노이즈 반영
            envelope = np.sin(np.pi * progress) # 0 -> 1 -> 0
            path_noise = envelope * mid_noise_amp * np.sin(mid_noise_freq * (t * 0.05) + mid_noise_phase)
            
            # 최종 전문가 목표 액션 (Action)
            target_ctrl = interpolated_pose + path_noise
            
            # 현재 물리 상태 및 제어 명령 기록
            all_states[ep, t] = data.qpos[:6].copy()
            all_actions[ep, t] = target_ctrl.copy()
            
            # 제어 신호 인가 후 물리 엔진 1스텝 전진
            data.ctrl[:6] = target_ctrl
            mujoco.mj_step(model, data)
            
        if (ep + 1) % 20 == 0 or ep == num_episodes - 1:
            print(f"  - 에피소드 [{ep+1:3d}/{num_episodes}] 수집 완료 (목표 도달 궤적)")

    # 4. 데이터 저장
    save_path = "demonstrations.npz"
    np.savez_compressed(
        save_path,
        states=all_states,      # Shape: [100, 100, 6]
        actions=all_actions     # Shape: [100, 100, 6]
    )
    
    print("\n✅ 특정 위치 1회 도달(Point-to-Point) 전문가 데모 수집 완료!")
    print(f"  - 시작 위치 (Base Start)  : {base_start_pose}")
    print(f"  - 목표 위치 (Base Target) : {base_target_pose}")
    print(f"  - 데이터셋 텐서 크기       : {all_states.shape} (에피소드 × 타임스텝 × 관절수)")
    print(f"  - 저장 파일 경로          : {os.path.abspath(save_path)}")
    print("=" * 60)

if __name__ == "__main__":
    collect_expert_demonstrations()
