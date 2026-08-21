# 📝 [실습 리포트 및 자가점검 답변] SO-101 6-DoF 시계열 모방학습(BC) with Vanilla RNN

---

## 1. 텐서 차원 흐름 (Tensor Flow) 검증 및 설명

학습 및 추론 루프에서 6축 관절 데이터가 각 신경망 레이어를 통과할 때의 텐서 Shape 변화는 다음과 같습니다:

```text
1. 입력 텐서 (X)              : [Batch=64, History=10, DoF=6]
       │
       ▼ nn.Linear(6, 64) + ReLU
2. 선형 투영 (Projection)     : [Batch=64, History=10, Embed_Dim=64]
       │
       ▼ nn.RNN(64, 128, batch_first=True)
3. RNN 순환 계층 (out)        : [Batch=64, History=10, Hidden_Dim=128]
       │
       ▼ out[:, -1, :] (마지막 타임스텝 추출)
4. 문맥 벡터 (Context)        : [Batch=64, Hidden_Dim=128]
       │
       ▼ nn.Linear(128, 6)
5. 출력 액션 헤드 (Y_pred)     : [Batch=64, DoF=6] (다음 1스텝 목표 관절각)
```

* **설명**: 
  - NLP의 `nn.Embedding`은 정수 단어 번호를 인덱싱(Lookup)하지만, **로봇공학에서는 관절 각도가 연속된 실수(Continuous Float)**이므로 `nn.Linear(6, 64)` 선형 투영 계층을 사용하여 6차원 물리 각도를 64차원의 풍부한 잠재 특징 공간(Latent Space)으로 확장합니다.
  - RNN 레이어는 $t-9$부터 $t$까지의 10스텝 연속 궤적을 순차 처리하여 관절의 속도/가속도/진행 방향 정보를 은닉 상태에 누적하며, 마지막 시점의 은닉 상태(`out[:, -1, :]`)가 시계열 전체를 압축 요약한 문맥 벡터가 됩니다.

---

## 2. 정량적 성능 평가 (Test MSE Loss)

학습에 참여하지 않은 **20개 독립 테스트 에피소드(총 1,800개 윈도우 샘플)**에 대한 평가 결과:

* **학습 설정**: Epoch 35, Batch Size 64, Adam Optimizer (lr=1e-3, weight_decay=1e-5), MSELoss
* **최종 Test MSE Loss**: **`0.063131`**
* **평가 고찰**: 35 Epoch 학습 과정에서 Train Loss는 지속적으로 감소하며, Test MSE Loss 역시 안정적으로 수렴하여 과적합(Overfitting) 없이 미학습 궤적에 대한 우수한 일반화 성능을 달성했습니다.

---

## 3. 단일 시점 MLP vs 시계열 RNN 비교 고찰

| 비교 항목 | 단일 시점 MLP 모델 ($q_t \rightarrow q_{t+1}$) | 시계열 RNN 모델 ($[q_{t-H+1}, \dots, q_t] \rightarrow q_{t+1}$) |
|---|---|---|
| **입력 데이터** | 현재 시점 1프레임 6차원 | 과거 $H=10$개 프레임 궤적 ($10 \times 6$) |
| **속도/가속도 인지** | 불가 (정적 위치만 관찰) | **암시적 속도/가속도/운동량 파악 가능** |
| **제어 동작의 부드러움** | **진동 및 떨림(Jittering) 발생 쉬움** | **시간적 일관성(Temporal Consistency)으로 매우 부드러움** |
| **외란(Noise) 저항성** | 센서 노이즈 1개에 즉시 출력 급변 | 과거 궤적 필터링 효과로 노이즈에 강건함 |

* **상세 고찰**:
  - 단일 시점 MLP는 로봇 팔이 현재 "어느 방향으로 얼마의 속도로 움직이고 있는지"를 알 수 없어 급격한 목표각 변화(High-frequency Jitter)가 발생하기 쉽습니다.
  - 반면 시계열 RNN은 과거 10개 프레임의 궤적 패턴을 통해 운동량과 위상(Phase)을 파악하므로, 시뮬레이터 Closed-loop 구동 시 오버슈트 없이 매끄러운 궤적 추종 성능을 보입니다.

---

## 4. 트랜스포머(Transformer)로의 확장 설계

현재의 바닐라 RNN 계층을 `nn.TransformerEncoder`로 교체할 때 필수적인 구성 요소와 그 이유는 다음과 같습니다:

1. **위치 인코딩 (Positional Encoding / Embedding)의 필수 도입**:
   - **이유**: RNN은 For-loop를 돌며 $t$번째를 순차적으로 계산하므로 시간 순서가 자연스럽게 반영되지만, Transformer의 Self-Attention은 모든 타임스텝($t-9 \sim t$)을 **동시에 병렬(Permutation Invariant)**로 처리합니다.
   - 따라서 입력 벡터에 "이 데이터가 $t-9$ 시점인지, 현재 $t$ 시점인지"를 알려주는 **Positional Encoding($PE$)**을 더해주지 않으면 시간 순서 정보가 완전히 유실됩니다.
2. **수식 구조**:
   $$\text{Transformer Input} = \text{Projector}(X)_{[B, H, E]} + \text{PositionalEncoding}_{[H, E]}$$
3. **Causal Mask (선택 사항)**:
   - 인코더-디코더 구조로 확장하여 미래 시점들을 autoregressive하게 예측할 때는 미래 정보를 보지 못하도록 가리는 Causal Attention Mask가 추가되어야 합니다.
