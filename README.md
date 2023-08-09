# AI-Trading-Project
증권사 API를 이용한 자동 매수/매도 시스템

## Description
빅데이터 기반 AI알고리즘 16기 1조 (피스메이커)

## Enviroment
<!DOCTYPE html>
<html>
<head>
    <title>텍스트 복사 예제</title>
</head>
<body>
    <p id="copyText">복사할 텍스트를 입력하세요.</p>
    <button onclick="copyToClipboard()">복사하기</button>

    <script>
        function copyToClipboard() {
            const textToCopy = document.getElementById("copyText").innerText;
            
            // 임시 입력 요소 생성
            const tempInput = document.createElement("textarea");
            tempInput.value = textToCopy;
            document.body.appendChild(tempInput);
            
            // 텍스트 선택 및 복사
            tempInput.select();
            document.execCommand("copy");
            
            // 임시 입력 요소 제거
            document.body.removeChild(tempInput);
            
            alert("텍스트가 복사되었습니다: " + textToCopy);
        }
    </script>
</body>
</html>


## 구성원
<table>
  <thead>
    <tr>
      <th>이름</th>
      <th>역할</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>정윤영 (팀장)</td>
      <td>주식 보조 지표, 강화학습 모델 구현</td>
    </tr>
    <tr>
      <td>김단비</td>
      <td>과거/실시간 주식 데이터 전처리, 강화학습 모델 구현</td>
    </tr>
    <tr>
      <td>민유진</td>
      <td>과거/실시간 뉴스 전처리 (자연어 처리) 후 코버트 모델 구현 후 적용 시킴</td>
    </tr>
    <tr>
      <td>박명윤</td>
      <td>클라우드, 리눅스 환경 생성</td>
    </tr>
    <tr>
      <td>박수연</td>
      <td>실시간/과거 주가 데이터 수집, DB연동, 데이터보관, 데이터 수집 자동화</td>
    </tr>
    <tr>
      <td>여주원</td>
      <td>CNN 모델 구현</td>
    </tr>
    <tr>
      <td>윤광현</td>
      <td>데이터를 64bit 환경으로 보내기 위한 레디스와 python 연동</td>
    </tr>
    <tr>
      <td>이예정</td>
      <td>과거/실시간 뉴스 크롤링, DB연동, 데이터보관, 웹 사이트 구현</td>
    </tr>
    <tr>
      <td>이창민</td>
      <td>자료조사, 모델링</td>
    </tr>
  </tbody>
</table>

 



## 나의 역할
CNN 아키텍처 설계와 구현, 주식 데이터의 전처리 및 변환, 모델 학습 및 평가 등
주식 데이터에 적용할 CNN(Convolutional Neural Network) 모델을 구현
주식 데이터는 시계열적인 특성을 가지며, 시간에 따라 변화하는 패턴을 포착하기 위해 CNN 적용
보조 지표 데이터 특성을 반영한 CNN 레이어를 설계하고, 주식 데이터를 시계열 형태로 변환하는 전처리

CNN은 이미지나 시계열 데이터에서 특징을 추출하고, 주가 데이터의 패턴을 학습하는 데 사용
CNN을 사용하여 주식 데이터 시계열 패턴 학습 후, 패턴 기반 강화학습으로 상승 및 하락 추세 예측
강화학습은 보조 지표나 기타 지표 데이터를 기반으로 주식 매매 결정, 강화학습 에이전트가 최적의 전략을 학습하도록 유도




