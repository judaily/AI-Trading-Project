# AI-Trading-Project
![프로젝트 아키텍처](https://github.com/judaily/AI-Trading-Project/blob/main/%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C.png)

## Description
빅데이터 기반 AI알고리즘 16기 1조 (피스메이커) - 증권사 API를 이용한 자동 매수/매도 시스템

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

데이터 특성을 고려한 CNN 설계하여 데이터를 시계열 형태로 전처리하고 주가 패턴 학습 <br> 
CNN으로 학습한 주식 데이터를 강화학습으로 패턴 기반한 상승 및 하락 추세 예측
