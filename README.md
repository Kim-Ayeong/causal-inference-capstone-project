# Causal Inference

## Experimental Data
- data
1. https://www.kaggle.com/datasets/adarsh0806/ab-testing-practice
2. https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing/data
3. https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats
- model: RCT, PSM 등

## Observational Data
- data: 2022~24년 대중교통 이용량(기후동행카드)
- model: DID, RD 등

## Result of Causal Model
- Numpy A/B testing ---> 결과: 대조 그룹(2,519명) 대비 처치 그룹(2,481명)에서 평균 time spent가 1.6 높으나 통계적으로 유의미하지 않음
- Marketing A/B testing ---> 결과: 대조 그룹(23,500명)의 2배로 처치 그룹(564,000명) 리샘플링 후, most_ads_day, most_ads_hour PSM 매칭으로 converted 차이 분석, ATE는 0.006로 통계적으로 유의미함
- 기후동행카드, DID ---> 결과: 2022/11/1~2024/10/31 동안 2024/1/27 기준, 부산 대비 서울의 시군구 대중교통 이용량 차이 분석, 전체 기간의 ATE는 8,741건
- 기후동행카드, RD ---> 결과: 2022/11/1~2024/10/31 동안 2024/1/27 기준, 서울에서의 시군구 평균 대중교통 이용량 차이 분석, 일별 ATE는 1,318건

## Dashboard
![image](https://github.com/user-attachments/assets/82eaf147-0bb6-42be-b212-834e79bf9473)
![image](https://github.com/user-attachments/assets/d552cb5e-1e59-464d-a3e6-76f05427af08)
![image](https://github.com/user-attachments/assets/67cc9b41-5038-4374-9966-9866a3b2318d)
![image](https://github.com/user-attachments/assets/1a6c0929-ee09-4fff-bf22-6b9e94a3c1e9)

