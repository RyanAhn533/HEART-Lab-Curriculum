import numpy as np
from sklearn.preprocessing import StandardScaler

#1. 데이터
data = np.array([[1,2,3, 1], 
                [4,5,6, 2],
                [7,8,9, 3],
                [10,11,12, 114],
                [13,14,15,115]])
print(data)

#1. 평균
means = np.mean(data, axis=0)
print('평균 : ',means )

#2. 모집단 분산 (n빵)  # ddof = 0 디폴트
#shape = 와꾸

population_variances = np.var(data, axis=0, )
print("모집단 분산 : ", population_variances)
#분산 :  [  18.   18.   18. 3038.]


#3. 표본분산 (n-1)빵  모표분분산보다 조금 더 큼
variances = np.var(data, axis=0, ddof=1)
print("표본 분산 : ", variances)
#분산 :  [  18.   18.   18. 3038.]

#4. 표본표준편차
std = np.std(data, axis=0, ddof=1)
print("표준편차 : ", std)
# 표준편차 :  [ 4.74341649  4.74341649  4.74341649 61.62385902]

#5. standardScaler
scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

print("standard_sclaer : \n", scaled_data)