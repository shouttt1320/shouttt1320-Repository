import os
import sys
import time
from collections import deque
import numpy as np
import torch
import mujoco
import mujoco.viewer
from step3_train_rnn import VanillaRNN_BC

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def run_closed_loop_evaluation(
    xml_path="scene.xml",
    model_path="best_rnn_bc_model.pth",
    sim_duration_sec=30.0,
    control_dt=0.05  # ⏱️ 제어 주기: 0.05초 (초당 20회 추론, 부드럽고 여유로운 속도)
):
    print("=" * 60)
    print("🤖 [Step 4] MuJoCo Closed-Loop 실시간 추론 제어 시작")
    print(f"   (제어 주기: {control_dt:.3f}초 - 여유롭고 부드러운 속도로 관찰 가능)")
    print("=" * 60)
    
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run Step 3 first!")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 모델 가중치 로드
    checkpoint = torch.load(model_path, map_location=device)
    history_len = checkpoint.get("history_len", 10)
    
    model = VanillaRNN_BC(
        input_dim=checkpoint.get("input_dim", 6),
        embed_dim=checkpoint.get("embed_dim", 64),
        hidden_dim=checkpoint.get("hidden_dim", 128),
        output_dim=checkpoint.get("output_dim", 6)
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  - 모델 로드 완료 (History={history_len}, Test MSE={checkpoint['test_loss']:.6f})")
    
    # 2. MuJoCo 시뮬레이터 초기화
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    
    # 데모 수집 시작 자세와 완벽히 일치하도록 초기화
    init_pose = np.array([-0.5, -0.40, 0.60, 0.30, 0.00, 0.10], dtype=np.float32)
    d.qpos[:6] = init_pose
    d.ctrl[:6] = init_pose
    mujoco.mj_step(m, d)
    
    # 3. 길이 H의 롤링 버퍼(Rolling Buffer) 초기화
    rolling_buffer = deque(maxlen=history_len)
    for _ in range(history_len):
        rolling_buffer.append(d.qpos[:6].copy())
        
    print("\n  - MuJoCo Passive 3D 뷰어를 실행합니다...")
    print("  - 실시간 Closed-Loop 추론 루프가 시작됩니다.")
    print("  - (창을 닫으면 시뮬레이션이 종료됩니다.)\n")
    
    start_real_time = time.time()
    step_count = 0
    
    # 1회 제어당 물리 연산 서브스텝 수 (물리 정밀도 500Hz 유지)
    n_substeps = max(1, int(control_dt / m.opt.timestep))
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # --- [Closed-Loop Pipeline] ---
            # 1. 롤링 버퍼 -> PyTorch 텐서 변환 [1, H, 6]
            buffer_array = np.array(rolling_buffer, dtype=np.float32) # [H, 6]
            input_tensor = torch.from_numpy(buffer_array).unsqueeze(0).to(device) # [1, H, 6]
            
            # 2. RNN 추론: 다음 1스텝 목표 관절각 예측
            with torch.no_grad():
                pred_action = model(input_tensor).squeeze(0).cpu().numpy() # [6]
                
            # 3. 예측값을 로봇 액추에이터에 인가
            d.ctrl[:6] = pred_action
            
            # 4. 물리 시뮬레이터 서브스텝 전진 (정밀 물리 계산)
            for _ in range(n_substeps):
                mujoco.mj_step(m, d)
            
            # 5. 새로운 물리 상태(data.qpos)를 버퍼에 갱신
            rolling_buffer.append(d.qpos[:6].copy())
            
            # 6. 화면 동기화
            viewer.sync()
            step_count += 1
            
            if step_count % 10 == 0:
                elapsed = time.time() - start_real_time
                print(f"[{elapsed:4.1f}s | Step {step_count:3d}] Pan: {pred_action[0]:+.2f}, Lift: {pred_action[1]:+.2f}, Elbow: {pred_action[2]:+.2f}, Wrist: {pred_action[3]:+.2f}, Roll: {pred_action[4]:+.2f}, Grip: {pred_action[5]:+.2f}")
                
            # 사람 눈에 알맞은 부드러운 속도로 프레임 딜레이 조절
            elapsed_in_step = time.time() - step_start
            time_until_next = control_dt - elapsed_in_step
            if time_until_next > 0:
                time.sleep(time_until_next)
                
            if (time.time() - start_real_time) > sim_duration_sec:
                break
                
    print("\n✅ Closed-Loop 평가 완료!")
    print("=" * 60)

if __name__ == "__main__":
    run_closed_loop_evaluation()
