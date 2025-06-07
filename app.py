import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go
from sklearn.cluster import MiniBatchKMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import cv2
import time

st.set_page_config(page_title="Interactive Semi-Supervised Learning", layout="wide")

# ================================
# 🎛️ 주요 매개변수 설정 (조절 가능)
# ================================

# 1. 모델 초기화 관련
MNIST_SAMPLES = 10000          # 사용할 MNIST 샘플 수
INITIAL_LABELS_PER_CLASS = 10   # 각 클래스별 초기 라벨 수
KMEANS_CLUSTERS = 50           # KMeans 클러스터 수
LABEL_PROP_NEIGHBORS = 15      # Label Propagation 이웃 수
LABEL_PROP_MAX_ITER = 200      # Label Propagation 최대 반복

# 2. 유사도 계산 가중치
COSINE_WEIGHT = 0.6           # Cosine similarity 가중치
CORRELATION_WEIGHT = 0.2      # Correlation 가중치
EUCLIDEAN_WEIGHT = 0.1        # Euclidean similarity 가중치
MANHATTAN_WEIGHT = 0.1        # Manhattan similarity 가중치

# 3. 업데이트 관련
NEAREST_NEIGHBORS = 50        # 고려할 가장 가까운 이웃 수
FORCED_UPDATE_COUNT = 5       # 무조건 업데이트할 상위 개수
SIMILARITY_THRESHOLD = 0.45   # 유사도 임계값
SIMILARITY_OVERWRITE = 0.5    # 덮어쓰기용 유사도 임계값
CONFIDENCE_THRESHOLD = 0.7    # 신뢰도 임계값
CONFIDENCE_CERTAIN = 0.99     # "확신 있는 예측" 기준

# 4. 디버깅
DEBUG_DISPLAY_COUNT = 10      # 디버깅 정보 표시 개수

# 5. 코인 시스템 관련
COIN_PER_ACCURACY_IMPROVE = 1  # 정확도 상승시 지급 코인
COIN_PER_CONSECUTIVE_FAIL = 1  # 3번 연속 하락시 위로 코인
CONSECUTIVE_FAIL_THRESHOLD = 3  # 연속 실패 임계값
DATA_PER_COIN = 100           # 1 코인당 구매 가능한 라벨 데이터 수

st.title("🧙‍♀️ 바보 모델 훈련소 & 데이터 상점")
st.markdown("""
**📚 옛날 옛날 아주 옛날에...**

숫자도 제대로 읽지 못하는 바보 모델이 있었습니다. 🤖💭  
이 불쌍한 모델은 겨우 몇 개의 손글씨만 본 적이 있어서, 새로운 숫자를 보면 늘 헷갈려했어요.

**당신의 임무:** 이 바보 모델에게 숫자를 가르쳐 주세요!  
✏️ 숫자를 그리고 → 🎯 모델이 틀렸다면 → 📖 정답을 알려주세요!

🪙 **코인 시스템:** 모델을 잘 가르치면 코인을 받아요!
- 정확도가 오르면 🪙 1코인
- 3번 연속 떨어져도 위로 🪙 1코인  
- 1코인으로 라벨 데이터 100개 구매 가능!

모델이 당신의 가르침을 받을 때마다 조금씩 똑똑해질 거예요. ✨
""")

# 세션 상태 초기화
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False
    st.session_state.accuracy_history = []
    st.session_state.labeled_count = 0
    
# 코인 시스템 초기화
if 'coins' not in st.session_state:
    st.session_state.coins = 0
    st.session_state.consecutive_fails = 0
    st.session_state.total_earned_coins = 0
    st.session_state.total_purchased_data = 0

def check_coin_reward(old_acc, new_acc):
    """코인 보상 체크 및 지급"""
    # 사용자 설정 가져오기
    settings = getattr(st.session_state, 'custom_settings', {})
    coin_improve = settings.get('COIN_PER_ACCURACY_IMPROVE', COIN_PER_ACCURACY_IMPROVE)
    coin_fail = settings.get('COIN_PER_CONSECUTIVE_FAIL', COIN_PER_CONSECUTIVE_FAIL)
    fail_threshold = settings.get('CONSECUTIVE_FAIL_THRESHOLD', CONSECUTIVE_FAIL_THRESHOLD)
    
    coins_earned = 0
    message = ""
    
    if new_acc > old_acc:
        # 정확도 상승시 코인 지급
        coins_earned += coin_improve
        st.session_state.consecutive_fails = 0  # 연속 실패 카운터 리셋
        message = f"🎉 정확도 상승! +{coin_improve} 🪙"
    elif new_acc < old_acc:
        # 정확도 하락시 연속 실패 카운터 증가
        st.session_state.consecutive_fails += 1
        if st.session_state.consecutive_fails >= fail_threshold:
            coins_earned += coin_fail
            st.session_state.consecutive_fails = 0  # 리셋
            message = f"😅 {fail_threshold}번 연속 하락... 위로 +{coin_fail} 🪙"
        else:
            remaining = fail_threshold - st.session_state.consecutive_fails
            message = f"📉 정확도 하락... ({remaining}번 더 떨어지면 위로 코인!)"
    else:
        # 동일한 정확도
        message = "➡️ 정확도 유지"
    
    if coins_earned > 0:
        st.session_state.coins += coins_earned
        st.session_state.total_earned_coins += coins_earned
        
    return coins_earned, message

def purchase_labeled_data():
    """코인으로 라벨 데이터 구매"""
    # 사용자 설정 가져오기
    settings = getattr(st.session_state, 'custom_settings', {})
    data_per_coin = settings.get('DATA_PER_COIN', DATA_PER_COIN)
    
    if st.session_state.coins < 1:
        return False, "코인이 부족합니다!"
    
    # 현재 unlabeled 데이터 찾기
    unlabeled_indices = np.where(st.session_state.labels == -1)[0]
    
    if len(unlabeled_indices) < data_per_coin:
        return False, f"구매 가능한 unlabeled 데이터가 {len(unlabeled_indices)}개뿐입니다!"
    
    # 랜덤하게 data_per_coin개 선택해서 라벨링
    selected_indices = np.random.choice(unlabeled_indices, data_per_coin, replace=False)
    for idx in selected_indices:
        st.session_state.labels[idx] = st.session_state.y[idx]  # 실제 정답으로 라벨링
    
    # 코인 차감
    st.session_state.coins -= 1
    st.session_state.total_purchased_data += data_per_coin
    st.session_state.labeled_count = np.sum(st.session_state.labels != -1)
    
    # 모델 재학습
    st.session_state.label_prop = LabelPropagation(
        kernel='knn', 
        n_neighbors=LABEL_PROP_NEIGHBORS,
        max_iter=LABEL_PROP_MAX_ITER
    )
    st.session_state.label_prop.fit(st.session_state.X, st.session_state.labels)
    
    # 새 정확도 계산
    new_acc = accuracy_score(
        st.session_state.y_test, 
        st.session_state.label_prop.predict(st.session_state.X_test)
    )
    st.session_state.accuracy_history.append(new_acc)
    
    return True, f"성공! {data_per_coin}개 데이터 구매 완료! 새 정확도: {new_acc:.1%}"

@st.cache_data
def load_mnist_fast():
    """빠른 MNIST 로드"""
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data[:MNIST_SAMPLES] / 255.0, mnist.target[:MNIST_SAMPLES].astype(int)
    return X, y

def initialize_model():
    # 데이터 로드
    X, y = load_mnist_fast()
    
    # 각 클래스별로 초기 라벨링
    labels = np.full(len(X), -1)
    for digit in range(10):
        digit_indices = np.where(y == digit)[0][:50]
        selected = np.random.choice(digit_indices, size=INITIAL_LABELS_PER_CLASS, replace=False)
        labels[selected] = digit
    
    # KMeans 클러스터링
    kmeans = MiniBatchKMeans(n_clusters=KMEANS_CLUSTERS, batch_size=500, random_state=42)
    kmeans.fit(X)
    
    # Label Propagation
    label_prop = LabelPropagation(kernel='knn', n_neighbors=LABEL_PROP_NEIGHBORS, max_iter=LABEL_PROP_MAX_ITER)
    label_prop.fit(X, labels)
    
    # 테스트 세트
    X_test, y_test = X[8000:], y[8000:]
    
    # 세션에 저장
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.labels = labels
    st.session_state.label_prop = label_prop
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.model_initialized = True
    st.session_state.labeled_count = INITIAL_LABELS_PER_CLASS * 10
    
    # 초기 정확도
    initial_acc = accuracy_score(y_test, label_prop.predict(X_test))
    st.session_state.accuracy_history = [initial_acc]

def preprocess_canvas(canvas_image):
    """캔버스 이미지를 MNIST 형식으로 변환"""
    if canvas_image is None:
        return None
    
    # RGB to Grayscale
    gray = cv2.cvtColor(canvas_image[:, :, :3], cv2.COLOR_RGB2GRAY)
    
    # 색상 반전: 흰 배경 + 검은 글씨 → 검은 배경 + 흰 글씨
    inverted = 255 - gray
    
    # 28x28로 리사이즈
    resized = cv2.resize(inverted, (28, 28))
    
    # 정규화
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized.reshape(1, -1)

def calculate_similarity_score(X, new_sample):
    """다양한 유사도 메트릭을 조합한 점수 계산"""
    # 사용자 설정 가져오기
    settings = getattr(st.session_state, 'custom_settings', {})
    cosine_w = settings.get('COSINE_WEIGHT', COSINE_WEIGHT)
    corr_w = settings.get('CORRELATION_WEIGHT', CORRELATION_WEIGHT)
    eucl_w = settings.get('EUCLIDEAN_WEIGHT', EUCLIDEAN_WEIGHT)
    manh_w = settings.get('MANHATTAN_WEIGHT', MANHATTAN_WEIGHT)
    
    # 1. Cosine similarity (각도 기반)
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sim = cosine_similarity(X, new_sample.reshape(1, -1)).flatten()
    
    # 2. Correlation (상관관계)
    correlations = []
    for i in range(len(X)):
        corr = np.corrcoef(X[i], new_sample.flatten())[0, 1]
        correlations.append(corr if not np.isnan(corr) else 0)
    correlations = np.array(correlations)
    
    # 3. Euclidean distance (거리 기반)
    euclidean_dist = np.sum((X - new_sample) ** 2, axis=1)
    euclidean_sim = 1 / (1 + euclidean_dist)  # 거리를 유사도로 변환
    
    # 4. Manhattan distance
    manhattan_dist = np.sum(np.abs(X - new_sample), axis=1)
    manhattan_sim = 1 / (1 + manhattan_dist)
    
    # 가중 조합 (사용자 설정 반영)
    combined_score = (
        cosine_w * cos_sim + 
        corr_w * correlations + 
        eucl_w * euclidean_sim + 
        manh_w * manhattan_sim
    )
    
    return combined_score

def quick_update(new_sample, true_label):
    start_time = time.time()
    
    # 사용자 설정 가져오기
    settings = getattr(st.session_state, 'custom_settings', {})
    nearest_count = settings.get('NEAREST_NEIGHBORS', NEAREST_NEIGHBORS)
    forced_count = settings.get('FORCED_UPDATE_COUNT', FORCED_UPDATE_COUNT)
    sim_threshold = settings.get('SIMILARITY_THRESHOLD', SIMILARITY_THRESHOLD)
    conf_threshold = settings.get('CONFIDENCE_THRESHOLD', CONFIDENCE_THRESHOLD)
    
    # 새 샘플 추가 전에 유사도 계산 (기존 데이터와)
    old_X = st.session_state.X.copy()
    similarity_scores = calculate_similarity_score(old_X, new_sample)
    
    # 유사도 높은 순으로 정렬 (내림차순)
    nearest_neighbors = np.argsort(similarity_scores)[::-1][:nearest_count]
    
    # 새 샘플을 데이터에 추가
    st.session_state.X = np.vstack([st.session_state.X, new_sample])
    st.session_state.labels = np.append(st.session_state.labels, true_label)
    
    # 이웃들의 신뢰도 계산 (기존 샘플들)
    neighbor_samples = old_X[nearest_neighbors]
    neighbor_proba = st.session_state.label_prop.predict_proba(neighbor_samples)
    neighbor_confidence = np.max(neighbor_proba, axis=1)
    
    updates_made = 0
    debug_info = []
    
    for i, idx in enumerate(nearest_neighbors):
        similarity = similarity_scores[idx]
        confidence = neighbor_confidence[i]
        current_label = st.session_state.labels[idx]
        
        debug_info.append(f"idx {idx}: similarity={similarity:.3f}, confidence={confidence:.3f}, label={current_label}")
        
        # 유사도 높은 순서대로 최소 개수는 무조건 업데이트
        if i < forced_count:
            st.session_state.labels[idx] = true_label
            updates_made += 1
            debug_info[-1] += f" -> FORCED_UPDATE(top{forced_count})"
        else:
            # 나머지는 조건부 업데이트
            high_similarity = similarity > sim_threshold
            low_confidence = confidence < conf_threshold
            
            if high_similarity and (current_label == -1 or low_confidence):
                if current_label != -1 and current_label != true_label:
                    # 높은 유사도일 때만 덮어쓰기
                    overwrite_threshold = settings.get('SIMILARITY_OVERWRITE', SIMILARITY_OVERWRITE)
                    if similarity > overwrite_threshold:
                        st.session_state.labels[idx] = true_label
                        updates_made += 1
                        debug_info[-1] += " -> UPDATED(overwrite)"
                else:
                    # unlabeled이거나 같은 라벨이면 업데이트
                    st.session_state.labels[idx] = true_label
                    updates_made += 1
                    debug_info[-1] += " -> UPDATED"
    
    # 재학습
    st.session_state.label_prop = LabelPropagation(
        kernel='knn', 
        n_neighbors=LABEL_PROP_NEIGHBORS,
        max_iter=LABEL_PROP_MAX_ITER
    )
    st.session_state.label_prop.fit(st.session_state.X, st.session_state.labels)
    
    # 새 정확도 계산
    old_acc = st.session_state.accuracy_history[-1] if st.session_state.accuracy_history else 0
    new_acc = accuracy_score(
        st.session_state.y_test, 
        st.session_state.label_prop.predict(st.session_state.X_test)
    )
    st.session_state.accuracy_history.append(new_acc)
    st.session_state.labeled_count = np.sum(st.session_state.labels != -1)
    
    # 코인 보상 체크
    coins_earned, coin_message = check_coin_reward(old_acc, new_acc)
    
    elapsed = time.time() - start_time
    
    # 전체 데이터 상태 분석
    unlabeled_count = np.sum(st.session_state.labels == -1)
    total_samples = len(st.session_state.labels)
    
    # 전체 신뢰도 분포 계산
    if total_samples > 100:  # 너무 많으면 샘플링
        sample_indices = np.random.choice(total_samples, 100, replace=False)
        sample_X = st.session_state.X[sample_indices]
        sample_proba = st.session_state.label_prop.predict_proba(sample_X)
        avg_confidence = np.mean(np.max(sample_proba, axis=1))
        low_conf_count = np.sum(np.max(sample_proba, axis=1) < CONFIDENCE_THRESHOLD)
    else:
        all_proba = st.session_state.label_prop.predict_proba(st.session_state.X)
        avg_confidence = np.mean(np.max(all_proba, axis=1))
        low_conf_count = np.sum(np.max(all_proba, axis=1) < CONFIDENCE_THRESHOLD)
    
    # 디버깅 정보 저장
    st.session_state.last_updates = updates_made
    st.session_state.debug_info = debug_info[:DEBUG_DISPLAY_COUNT]
    st.session_state.data_stats = {
        'unlabeled_count': unlabeled_count,
        'total_samples': total_samples,
        'avg_confidence': avg_confidence,
        'low_conf_count': low_conf_count
    }
    st.session_state.last_coin_message = coin_message
    st.session_state.last_coins_earned = coins_earned
    
    return new_acc, elapsed

# 자동 모델 초기화
if not st.session_state.model_initialized:
    with st.spinner("🚀 모델 초기화 중... (약 10초)"):
        initialize_model()
    st.success("✅ 모델 초기화 완료!")
    st.rerun()
else:
    # 상단 코인 정보는 데이터 상점으로 통합
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("✏️ 모델에게 숫자 보여주기")
        
        # 캔버스 스타일 개선 (다크모드 대응)
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=15,
            stroke_color="#000000", 
            background_color="#FFFFFF",
            update_streamlit=True,
            height=200,
            width=200,
            drawing_mode="freedraw",
            point_display_radius=0,
            display_toolbar=True,
            key="canvas"
        )
        
        
        if st.button("🎯 모델아, 이게 뭐야?"):
            # 플래그 제거
                
            if canvas_result.image_data is not None:
                image = preprocess_canvas(canvas_result.image_data)
                if image is not None:
                    pred = st.session_state.label_prop.predict(image)[0]
                    proba = st.session_state.label_prop.predict_proba(image)[0]
                    confidence = np.max(proba)
                    
                    st.session_state.current_pred = pred
                    st.session_state.current_conf = confidence
                    st.session_state.current_image = image
    
    with col2:
        st.subheader("🤖 바보 모델의 대답")
        
        # 구매 성공 메시지 제거 (한번만 보이게)
        
        if 'current_pred' in st.session_state:
            st.metric("모델의 답", f"'{st.session_state.current_pred}'", f"확신도: {st.session_state.current_conf:.1%}")
            
            if st.session_state.current_conf < CONFIDENCE_CERTAIN:
                st.warning("😅 으음... 잘 모르겠어요!")
                st.markdown("**🧙‍♀️ 정답을 가르쳐 주세요:**")
                
                # 숫자 버튼들 (5개씩 2줄)
                col1_btn, col2_btn, col3_btn, col4_btn, col5_btn = st.columns(5)
                col6_btn, col7_btn, col8_btn, col9_btn, col10_btn = st.columns(5)
                
                button_pressed = None
                
                with col1_btn:
                    if st.button("0", use_container_width=True):
                        button_pressed = 0
                with col2_btn:
                    if st.button("1", use_container_width=True):
                        button_pressed = 1
                with col3_btn:
                    if st.button("2", use_container_width=True):
                        button_pressed = 2
                with col4_btn:
                    if st.button("3", use_container_width=True):
                        button_pressed = 3
                with col5_btn:
                    if st.button("4", use_container_width=True):
                        button_pressed = 4
                with col6_btn:
                    if st.button("5", use_container_width=True):
                        button_pressed = 5
                with col7_btn:
                    if st.button("6", use_container_width=True):
                        button_pressed = 6
                with col8_btn:
                    if st.button("7", use_container_width=True):
                        button_pressed = 7
                with col9_btn:
                    if st.button("8", use_container_width=True):
                        button_pressed = 8
                with col10_btn:
                    if st.button("9", use_container_width=True):
                        button_pressed = 9
                
                # 버튼이 눌렸을 때 학습 실행
                if button_pressed is not None:
                    with st.spinner("학습 중..."):
                        old_acc = st.session_state.accuracy_history[-1]
                        new_acc, elapsed = quick_update(st.session_state.current_image, button_pressed)
                        
                        improvement = (new_acc - old_acc) * 100
                        
                        # 정확도 변화를 세션에 저장 (col3에서 표시용)
                        st.session_state.last_accuracy_change = improvement
                        # 플래그 제거
                        
                        st.success(f"🎉 고마워요! '{button_pressed}'를 {elapsed:.1f}초 만에 배웠어요!")
                        
                        # 코인 보상 메시지만 표시 (정확도 관련 메시지는 col3으로 이동)
                        if hasattr(st.session_state, 'last_coin_message'):
                            if st.session_state.last_coins_earned > 0:
                                st.success(st.session_state.last_coin_message)
                            else:
                                st.info(st.session_state.last_coin_message)
                        
                        # 디버깅 정보는 아래 로그로 이동
            else:
                st.success("😎 이건 자신 있어요! 맞죠?")
    
    with col3:
        st.subheader("📈 모델의 성장 일기")
        
        # 현재 통계
        current_acc = st.session_state.accuracy_history[-1]
        
        # 정확도 변화 표시 (하나만)
        if hasattr(st.session_state, 'last_accuracy_change'):
            change = st.session_state.last_accuracy_change
            if change > 0:
                st.metric("정확도 변화", f"{current_acc:.1%}", f"+{change:.2f}%p")
            elif change < 0:
                st.metric("정확도 변화", f"{current_acc:.1%}", f"{change:.2f}%p")
            else:
                st.metric("정확도 변화", f"{current_acc:.1%}", f"±0.00%p")
        else:
            st.metric("현재 정확도", f"{current_acc:.1%}")
        
        st.metric("라벨링된 샘플", f"{st.session_state.labeled_count}")
        
        # 정확도 변화 그래프
        if len(st.session_state.accuracy_history) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.accuracy_history,
                mode='lines+markers',
                name='정확도'
            ))
            fig.update_layout(
                title="실시간 성능 향상",
                xaxis_title="업데이트 횟수",
                yaxis_title="정확도",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 데이터 상점 & 코인 정보 통합
        st.markdown("---")
        st.markdown("### 🏪 데이터 상점")
        
        # 코인 현황 (간략화)
        unlabeled_count = np.sum(st.session_state.labels == -1)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🪙 보유 코인", st.session_state.coins)
        with col2:
            st.metric("🏷️ 구매 가능 데이터", f"{unlabeled_count:,}")
        
        # 구매 섹션
        settings = getattr(st.session_state, 'custom_settings', {})
        current_data_per_coin = settings.get('DATA_PER_COIN', DATA_PER_COIN)
        
        if st.button(f"🛒 데이터 {current_data_per_coin}개 구매 및 공부 (1🪙)", 
                    disabled=st.session_state.coins < 1 or unlabeled_count < current_data_per_coin,
                    use_container_width=True):
            success, message = purchase_labeled_data()
            if success:
                # 구매 성공 시 바로 새로고침 (메시지 저장 안함)
                st.rerun()
            else:
                st.error(message)

# ================================
# 📋 시스템 로그 (개발자용)
# ================================

with st.expander("📋 시스템 로그 (고급 사용자용)", expanded=False):
    if hasattr(st.session_state, 'last_updates'):
        st.write(f"🔄 최근 업데이트: {st.session_state.last_updates}개 샘플")
    
    if hasattr(st.session_state, 'debug_info'):
        st.subheader("🔍 디버깅 정보")
        log_text = "\n".join(st.session_state.debug_info)
        st.text_area("가까운 이웃들", log_text, height=150)
    
    if hasattr(st.session_state, 'data_stats'):
        stats = st.session_state.data_stats
        st.subheader("📊 데이터 상태")
        st.text(f"Unlabeled 샘플: {stats['unlabeled_count']}/{stats['total_samples']}")
        st.text(f"평균 신뢰도: {stats['avg_confidence']:.3f}")
        st.text(f"낮은 신뢰도(<0.8) 샘플: {stats['low_conf_count']}")

# ================================
# 🔧 숨겨진 고급 설정 (전문가용)
# ================================

st.markdown("---")
st.markdown("## ⚙️ 고급 설정 (전문가용)")
with st.expander("🔧 매개변수 조절하기", expanded=False):
    st.markdown("### 🎯 유사도 계산 가중치")
    st.caption("합계가 1.0이 되도록 조절하세요")
    
    cosine_w = st.slider("Cosine Similarity", 0.0, 1.0, COSINE_WEIGHT, 0.1, key="cosine")
    corr_w = st.slider("Correlation", 0.0, 1.0, CORRELATION_WEIGHT, 0.1, key="corr") 
    eucl_w = st.slider("Euclidean Distance", 0.0, 1.0, EUCLIDEAN_WEIGHT, 0.1, key="eucl")
    manh_w = st.slider("Manhattan Distance", 0.0, 1.0, MANHATTAN_WEIGHT, 0.1, key="manh")
    
    total_weight = cosine_w + corr_w + eucl_w + manh_w
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"⚠️ 가중치 합계: {total_weight:.2f} (권장: 1.0)")
    
    st.markdown("### 📊 업데이트 설정")
    neighbors = st.slider("고려할 이웃 수", 10, 100, NEAREST_NEIGHBORS, 5, key="neighbors")
    forced = st.slider("강제 업데이트 수", 1, 10, FORCED_UPDATE_COUNT, 1, key="forced")
    sim_thresh = st.slider("유사도 임계값", 0.1, 0.9, SIMILARITY_THRESHOLD, 0.05, key="sim_thresh")
    conf_thresh = st.slider("신뢰도 임계값", 0.1, 0.9, CONFIDENCE_THRESHOLD, 0.05, key="conf_thresh")
    
    st.markdown("### 🎚️ 모델 설정")
    prop_neighbors = st.slider("Label Propagation 이웃 수", 5, 30, LABEL_PROP_NEIGHBORS, 1, key="prop_neighbors")
    
    st.markdown("### 🪙 코인 시스템 설정")
    coin_accuracy = st.slider("정확도 상승 보상 코인", 1, 5, COIN_PER_ACCURACY_IMPROVE, 1, key="coin_accuracy")
    coin_fail = st.slider("연속 하락 위로 코인", 1, 5, COIN_PER_CONSECUTIVE_FAIL, 1, key="coin_fail")
    fail_threshold = st.slider("연속 하락 임계값", 2, 5, CONSECUTIVE_FAIL_THRESHOLD, 1, key="fail_threshold")
    data_per_coin = st.slider("1코인당 데이터 개수", 50, 200, DATA_PER_COIN, 50, key="data_per_coin")
    
    if st.button("🔄 설정 적용", type="primary"):
        # 글로벌 변수 업데이트 (주의: 이건 권장되지 않지만 데모용)
        st.session_state.custom_settings = {
            'COSINE_WEIGHT': cosine_w,
            'CORRELATION_WEIGHT': corr_w, 
            'EUCLIDEAN_WEIGHT': eucl_w,
            'MANHATTAN_WEIGHT': manh_w,
            'NEAREST_NEIGHBORS': neighbors,
            'FORCED_UPDATE_COUNT': forced,
            'SIMILARITY_THRESHOLD': sim_thresh,
            'CONFIDENCE_THRESHOLD': conf_thresh,
            'LABEL_PROP_NEIGHBORS': prop_neighbors,
            'COIN_PER_ACCURACY_IMPROVE': coin_accuracy,
            'COIN_PER_CONSECUTIVE_FAIL': coin_fail,
            'CONSECUTIVE_FAIL_THRESHOLD': fail_threshold,
            'DATA_PER_COIN': data_per_coin
        }
        st.success("✅ 설정이 다음 학습부터 적용됩니다!")
        
    if st.button("🔙 기본값 복원"):
        if 'custom_settings' in st.session_state:
            del st.session_state.custom_settings
        st.success("✅ 기본 설정으로 복원되었습니다!")
        st.rerun()
    
    # 현재 설정 표시
    st.markdown("### 📋 현재 적용 중인 값")
    current_settings = getattr(st.session_state, 'custom_settings', {})
    st.caption(f"강제 업데이트: {current_settings.get('FORCED_UPDATE_COUNT', FORCED_UPDATE_COUNT)}개")
    st.caption(f"유사도 임계값: {current_settings.get('SIMILARITY_THRESHOLD', SIMILARITY_THRESHOLD)}")
    st.caption(f"신뢰도 임계값: {current_settings.get('CONFIDENCE_THRESHOLD', CONFIDENCE_THRESHOLD)}")
    st.caption(f"정확도 상승 코인: {current_settings.get('COIN_PER_ACCURACY_IMPROVE', COIN_PER_ACCURACY_IMPROVE)}🪙")
    st.caption(f"연속 하락 코인: {current_settings.get('COIN_PER_CONSECUTIVE_FAIL', COIN_PER_CONSECUTIVE_FAIL)}🪙")
    st.caption(f"1코인당 데이터: {current_settings.get('DATA_PER_COIN', DATA_PER_COIN)}개")