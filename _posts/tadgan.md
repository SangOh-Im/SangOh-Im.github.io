Paper URL: 
https://arxiv.org/pdf/2009.07769.pdf
Source : 
https://github.com/sintel-dev/Orion
https://github.com/ds-academy/TadGAN

TadGAN은 비지도 기반 시계열 이상 탐지 모델로 IEEE Big Data 2020에서 발표되었다. 이곳에서 연구팀은 아래와 같은 말을 한다.

TadGAN 알고리즘이 언젠가 위성 회사 뿐만 아니라 다양한 산업에 서비스를 제공하기를 희망합니다. 예를 들어, TadGAN은 줌(Zoom)과 같은 회사가 데이터 센터의 시계열 신호(예: CPU 사용량 또는 온도)를 모니터링하여 서비스 중단을 방지하도록 도울 수 있습니다.

이를 위해 연구팀은 현재까지 TadGAN 코드를 정기적으로 업로드하고 있다. 좋은 성능 뿐만 아니라 재현성까지 보장된다고 판단하여 해당 알고리즘에 대한 연구를 시작했다.

개요

대표적인 시계열 이상 탐지 방법은 아래와 같다.




out of limit mehtod는 일정 값을 넘으면 이상으로 간주하는 가장 단순한 방법이다. 하지만 contextual outlier와 같이 해당 구간에서만 이상에 해당하는 경우 탐지하지 못한다.
Proximity는 먼저 객체 사이의 거리를 통해 데이터 각각의 이상 정도를 수치화한다. 해당 방법은 거리 기반 알고리즘과 밀도 기반 알고리즘으로 구분된다. 이 방법은 두 가지 문제점이 있다. 먼저, 이상의 개수와 같은  파라미터를 설정할 필요가 있다. 이러한 설정한 충분한 도메인 지식이 요구된다. 두 번째 문제점은 시간적인 상관 관계를 잡아낼 수 없다는 것이다. 
Predict는 먼저 미래의 값을 예측한 후 이 값이 실제와 차이가 날 경우 이상으로 판단하는 알고리즘이다. ARIMA를 포함한 다양한 통계 기법들이 존재한다. 하지만 이러한 통계 기법들은 파라미터의 선택에 매우 민감하고 모델 설계에 많은 가정이 필요하다. 따라서 LSTM을 포함한 딥러닝 방법들의 활발하게 연구되고 있다.
Recontruction은 주어진 시계열의 latent space를 잡아낸다. 재구축 기반 방법론은 이상 데이터의 경우 저차원으로 압축될 때 많은 양의 정보를 잃게 되어 결과적으로 높은 재구축 오차가 발생한다고 가정한다. AE, VAE 등이 여기에 해당된다. 해당 모델은 학습 능력이 뛰어나 이상치 역시 학습하는 문제점을 가지고 있다. 
Reconstruction(GANs)는 Generator와 Discriminator가 경쟁적으로 학습하는 모델이다. GANs, BEATGAN 등이 대표적이다. GAN 알고리즘은 일반적으로 데이터의 특성을 잘 잡지 못한다. 따라서 학습이 잘 이루어지지 않는 문제점이 있다. GAN의 또 다른 문제점은 Mode Collapse이다.  Mode Collapse는 아래 그림과 같이 생성자가 다양한 데이터를 생성하지 않는 문제를 말한다.





TadGAN 알고리즘은 GAN에 기반한 앙상블 형태의 알고리즘이다. 아래는 TadGAN의 특성이다.

GAN과 AutoEncoder의 결합을 통해 적당한 수준의 이상을 탐지할 수 있도록 균형을 맞추었다.
추가적인 loss 활용하여 Mode Collapse 문제를 완화했다.




알고리즘 소개

입력 데이터(X)의 형태는 아래와 같이 길이가 100인 subsequence이다. 해당 subsequence가 encoder를 지나면 코딩 영역에서 길이가 20인 subsequence(Z)가 된다. Z가 다시 generator를 지나면 길이가 100인 subsequence로 재구축된다.




TadGAN 알고리즘 도식화


Critic X
Critic Z 


Encoder
Generator








각 신경망의 미션
Cx : 실제 데이터(X)와 잡음을 통해 생성한 가짜 데이터(g(z))를 구분한다.
Cz : 실제 데이터를 압축한 데이터(E(x))와 잡음을 구분한다.
E : 고차원의 subsequence를 저차원으로 압축한다.
G : 저차원의 subsequence를 고차원으로 디코딩한다.


손실 함수

Wasserstein Loss
두 분포 간의 차이를 나타내는 loss이다. wasserstein loss는 모드 붕괴를 완화할 수 있는 손실 함수로 평가 받고있다. 



Wasserstein Distance는 두 분포 사이의 거리를 의미한다. 위의 그림은 두 분포 Pr과 Pg의 Wasserstein Distance를 구한 것이다. 이는 블록을 움직이는 것과 유사하다. Pr에 해당하는 블록들을 Pg로 옮긴다고 생각하면 다양한 방법들이 있을 것이다. 이것을 joint distribution이라고 한다. 이 중에 가장 거리가 가까운 것을 Wasserstein Distance라고 부른다. 위의 경우 블록(분포)이 6개이므로 Wasserstein Distance는 18/6=3이다.



joint distribution의 수가 많아 질 경우 해당 방법은 계산에 상당한 부담이 간다. 따라서 위의 방법을 수학적 기법을 통해 max 문제로 치환한다. 위의 두 식은 구조는 다르지만 같은 의미를 가지고 있다. 여기서 fw 함수는 1-Lipschitz 연속 함수를 의미한다. 1-Lipschitz 연속 함수는 함수를 매끈하게 만들어 gradient explosion의 위험을 줄여 학습이 안정적으로 이루어진다. 


2. Cycle Consistency Loss
 일반적인 AutoEncoder의 loss(L2 loss)에 해당한다. GAN loss(i.e. Wasserstein Loss)만 이용할 경우 reconstruct가 잘 이루어질 가능성이 낮다. 따라서 Cycle Consistency loss 추가를 통해 이러한 위험을 낮추었다.

3. Full Object 



학습 과정

일반적인 GAN 학습과 같이 모델 한 개가 학습되는 동안 다른 모델 내부의 가중치는 고정되어있다. 아래는 학습의 순서이다.
Critic x ➝ Critic z ➝ Encoder ➝ Decoder

Critic x가 학습될 때를 자세히 설명해보면 학습이 이루어지는 동안 다른 모델인 Critic z, Encoder, Decoder의 가중치는 고정된다.  이 상태에서 아래의 손실 함수가 작아지는 방향으로 학습이 진행된다.

@tf.function
def Critic x 손실함수(x, z):
  with tf.GradientTape() as tape:

    valid_x = critic_x(x)
    x_ = generator(z)
    fake_x = critic_x(x_)

    # Interpolated 
    alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
    interpolated = alpha * x + (1 - alpha) * x_ 

    with tf.GradientTape() as gp_tape:
      gp_tape.watch(interpolated)
      pred = critic_x(interpolated)

    grads = gp_tape.gradient(pred, interpolated)
    grad_norm = tf.norm(tf.reshape(grads, (batch_size, -1)), axis=1)
    gp_loss = 10.0*tf.reduce_mean(tf.square(grad_norm - 1.))

    loss1 = wasserstein_loss(-tf.ones_like(valid_x), valid_x)
    loss2 = wasserstein_loss(tf.ones_like(fake_x), fake_x)

    loss = loss1 + loss2 + gp_loss


  gradients = tape.gradient(loss, critic_x.trainable_weights)  
  critic_x_optimizer.apply_gradients(zip(gradients, critic_x.trainable_weights))

  return loss


loss1 : Critic x가 진짜 데이터를 진짜 데이터로 판단하는 능력을 나타냄.
loss2 : Critic x가 가짜 데이터를 가짜 데이터로 판단하는 능력을 나타냄.
gp_loss : 실제 데이터와 가짜 데이터의 유사한 정도를 나타냄.

total loss = loss1 + loss2 + gp_loss




Anomaly Scoring



논문에서는 anomaly score 계산하는 몇 가지 방법을 제공한다. multiply 방식의 경우 
anomaly score = Critic score * Reconstruct score 의 형태로 결정된다.


Critic score
Critic x는 해당 데이터가 실제 데이터인지 가짜 데이터인지 판단하는 모델임을 위에서 소개했다. Critic x가 0에 가까우면 실제 데이터이고 크기가 커질 수록 가짜 데이터임을 나타낸다. 따라서 이 Critic x의 값을 anomaly scoring에 그대로 이용할 수 있다. 논문에서는 DTW(rec score의 일종)와 Critic의 결과를 곱한 anomaly scoring 기법이 가장 좋았다고 말한다. 

Reconstruct score의 3가지 Type

단순히 실제 데이터와 복원한 데이터 사이의 거리를 통한 point 방법부터 영역의 이상을 고려할 수 있는 Area와 DTW방법에 대해 소개한다. 

point 


Area


DTW




다음은 TadGAN을 통해 anomaly score(붉은 그래프)를 구한 것이다. 푸른색 그래프가 실제 데이터, 노란색 그래프가 재구축 데이터이다.



Find Anomaly 

TadGAN은 thredshold 결정 로직을 제공한다. 간단하게 소개하면 anomaly score를 subsequence로 분리한다.  그 후 subsequence마다 4sigma 넘어갈 경우 이상으로 결정한다. 이후, merge하여 최종 이상을 탐지한다. 아래는 TadGAN 로직에 따른 이상 탐지(붉은 부분) 형태이다. 실제 이상으로 라벨링 된 부분은 녹색으로 표시하여 서로 비교해보았다.




