# 필요한 라이브러리 임포트
import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn import metrics

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows 기준)
font_path = r'C:\Windows\Fonts\NanumGothic.ttf'
if os.path.exists(font_path):
    import matplotlib.font_manager as fm

    fe = fm.FontEntry(fname=font_path, name='NanumGothic')
    fm.fontManager.ttflist.insert(0, fe)
    plt.rcParams.update({'font.size': 10, 'font.family': 'NanumGothic'})
else:
    print("폰트 파일을 찾을 수 없습니다. 경로를 확인하세요.")

# 데이터 경로 설정
train_path = 'data/train.csv'
test_path = 'data/test.csv'
bus_feature_path = 'data/bus_feature.csv'
subway_feature_path = 'data/subway_feature.csv'
complete_path = 'data/output_final_v2.csv'
interest_rate_path = 'data/interest_rate.csv'

# 데이터 불러오기
df = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
bus = pd.read_csv(bus_feature_path)
subway = pd.read_csv(subway_feature_path)
interest_rate = pd.read_csv(interest_rate_path)
concat_df = pd.read_csv(complete_path)



# 불필요한 공백이나 '-'을 결측치로 대체
concat_df['등기신청일자'] = concat_df['등기신청일자'].replace(' ', np.nan)
concat_df['거래유형'] = concat_df['거래유형'].replace('-', np.nan)
concat_df['중개사소재지'] = concat_df['중개사소재지'].replace('-', np.nan)

# 결측치가 많은 열 제외 (100만개 이하 결측치 허용)
selected = list(concat_df.columns[concat_df.isnull().sum() <= 1000000])
concat_select = concat_df[selected]
print(concat_select.shape)
# 범주형 변수로 변환
concat_select['본번'] = concat_select['본번'].astype('str')
concat_select['부번'] = concat_select['부번'].astype('str')

# 연속형과 범주형 변수 분리
continuous_columns = [col for col in concat_select.columns if pd.api.types.is_numeric_dtype(concat_select[col])]
categorical_columns = [col for col in concat_select.columns if col not in continuous_columns]

# 결측치 처리
concat_select.loc[:, categorical_columns] = concat_select[categorical_columns].fillna('NULL')
concat_select.loc[:, continuous_columns] = concat_select[continuous_columns].interpolate(method='linear', axis=0)

# 계약년월 분할
concat_select['계약년'] = concat_select['계약년월'].astype(str).str[:4]
concat_select['계약월'] = concat_select['계약년월'].astype(str).str[4:]
concat_select.drop('계약년월', axis=1, inplace=True)

interest_rate = interest_rate.drop(['통계표', '계정항목', '단위', '변환'], axis=1)
interest_rate_melted = interest_rate.melt(var_name='계약년', value_name='금리')
concat_select = pd.merge(concat_select, interest_rate_melted, on='계약년', how='left')

# 강남 여부 파생 변수 추가
gangnam = ['강서구', '영등포구', '동작구', '서초구', '강남구', '송파구', '강동구']
concat_select['강남여부'] = concat_select['구'].isin(gangnam).astype(int)

# 신축 여부 파생 변수 추가
concat_select['신축여부'] = (concat_select['건축년도'] >= 2009).astype(int)

# 원-핫 인코딩으로 변환할 변수 정의
onehot_encode_cols = ['기타/의무/임대/임의=1/2/3/4']

# 원-핫 인코딩 적용
concat_select = pd.get_dummies(concat_select, columns=onehot_encode_cols, prefix=onehot_encode_cols, drop_first=True)

# 빈도 인코딩 적용할 변수 정의 (원-핫 인코딩된 변수 제외)
freq_encode_cols = ['구', '동', 'k-단지분류(아파트,주상복합등등)', 'k-세대타입(분양형태)',
                    'k-관리방식', 'k-복도유형', 'k-난방방식', '경비비관리형태', '세대전기계약방법',
                    '청소비관리형태', '강남여부', '신축여부']

for col in freq_encode_cols:
    concat_select[col] = concat_select[col].map(concat_select[col].value_counts(normalize=True))

# 타겟 인코딩
for col in ['도로명', '아파트명', 'k-시행사', 'k-건설사(시공사)']:
    means = concat_select.groupby(col)['target'].mean()
    concat_select[f'{col}_타겟인코딩'] = concat_select[col].map(means)

# 타겟 인코딩된 열 삭제
concat_select.drop(['도로명', '아파트명', 'k-시행사', 'k-건설사(시공사)'], axis=1, inplace=True)

# 타겟 인코딩된 열 스케일링 (트리 모델에서는 생략 가능)
target_encoded_cols = ['도로명_타겟인코딩', '아파트명_타겟인코딩',
                       'k-시행사_타겟인코딩', 'k-건설사(시공사)_타겟인코딩']
scaler = MinMaxScaler()
concat_select[target_encoded_cols] = scaler.fit_transform(concat_select[target_encoded_cols])

# 이진 인코딩
binary_cols = ['사용허가여부', '관리비 업로드']
for col in binary_cols:
    concat_select[col] = concat_select[col].map({'Y': 1, 'N': 0})

# 날짜 형식 변환
date_cols = ['k-사용검사일-사용승인일', 'k-수정일자', '단지승인일', '단지신청일']
for col in date_cols:
    concat_select[col] = pd.to_datetime(concat_select[col], errors='coerce')

# 식별자 관련 변수 삭제
identifier_cols = ['k-전화번호', 'k-팩스번호', '고용보험관리번호']
concat_select.drop(identifier_cols, axis=1, inplace=True)
print(concat_select.shape)
# 정규성 검토를 위한 연속형 변수 정리
continuous_columns = ['전용면적(㎡)', '계약년', '계약월', '층', '건축년도', 'k-전체동수', 'k-전체세대수',
                      'k-연면적', 'k-주거전용면적', 'k-전용면적별세대현황(60㎡이하)',
                      'k-전용면적별세대현황(60㎡~85㎡이하)', '건축면적', '주차대수', '좌표X', '좌표Y',
                      '가장 가까운 지하철역까지의 거리', '가장 가까운 버스 정류장까지의 거리',
                      '500m 이내 지하철역 수', '500m 이내 버스 정류장 수', '금리', '계약일']

# 숫자형으로 변환
for col in continuous_columns:
    concat_select[col] = pd.to_numeric(concat_select[col], errors='coerce')

# IQR 방식으로 이상치 제거 함수 정의
def remove_outliers_iqr(df, column):
    train_df = df[df['is_test'] == 0]
    test_df = df[df['is_test'] == 1]
    Q1 = train_df[column].quantile(0.25)
    Q3 = train_df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    train_df = train_df[(train_df[column] >= lower) & (train_df[column] <= upper)]
    return pd.concat([train_df, test_df])

# 이상치 제거 적용
concat_select = remove_outliers_iqr(concat_select, '전용면적(㎡)')
print(concat_select.shape)
# 연속형 변수 스케일링 (트리 모델에서는 생략 가능)
scaler = MinMaxScaler()
concat_select[continuous_columns] = scaler.fit_transform(concat_select[continuous_columns])

# 상관관계가 높은 변수 삭제
cols_to_drop = [
    'k-관리비부과면적', 'geometry', '번지', '본번', '부번',
    'k-사용검사일-사용승인일', 'k-수정일자', '단지승인일', '단지신청일',
    'k-85㎡~135㎡이하', 'k-전체세대수', 'k-주거전용면적',
    '건축년도', 'k-연면적', 'k-시행사_타겟인코딩',
    'k-건설사(시공사)_타겟인코딩', '강남여부', '동',
    '청소비관리형태', '세대전기계약방법', 'k-복도유형',
    'k-난방방식', '경비비관리형태', 'k-관리방식'
]
concat_select.drop(cols_to_drop, axis=1, inplace=True)




# 데이터 확인
print(tabulate(concat_select.head(5), headers='keys', tablefmt='fancy_grid', showindex=True))
print(concat_select.shape)
"""
모델링 (선생님 코드 사용)
"""

# 학습용 데이터와 테스트 데이터 분리
dt_train = concat_select.query('is_test==0').drop(['is_test'], axis=1)
dt_test = concat_select.query('is_test==1').drop(['is_test'], axis=1)
print(dt_train.shape, dt_test.shape)

# 테스트 데이터의 target을 임의로 설정
dt_test['target'] = 0

# 연속형과 범주형 변수 재분리
continuous_columns_v2 = [col for col in dt_train.columns if pd.api.types.is_numeric_dtype(dt_train[col])]
categorical_columns_v2 = [col for col in dt_train.columns if col not in continuous_columns_v2]

# 원-핫 인코딩된 변수 제외한 나머지 범주형 변수 리스트 정의
# '기타/의무/임대/임의=1/2/3/4'는 이미 원-핫 인코딩 되었으므로 제외
remaining_categorical_cols = [col for col in categorical_columns_v2 if col not in onehot_encode_cols]

# 레이블 인코딩
label_encoders = {}
for col in tqdm(remaining_categorical_cols, desc="레이블 인코딩 진행"):
    lbl = LabelEncoder()
    dt_train[col] = dt_train[col].astype(str).fillna('NULL')
    dt_test[col] = dt_test[col].astype(str).fillna('NULL')

    # 'NULL'을 포함하도록 classes_에 추가
    unique_train = dt_train[col].unique().tolist()
    if 'NULL' not in unique_train:
        unique_train.append('NULL')
    lbl.fit(unique_train)

    dt_train[col] = lbl.transform(dt_train[col])
    label_encoders[col] = lbl

    # 테스트 데이터에 새로운 레이블 처리
    dt_test[col] = dt_test[col].apply(lambda x: x if x in lbl.classes_ else 'NULL')
    dt_test[col] = lbl.transform(dt_test[col])

assert dt_train.shape[1] == dt_test.shape[1]

# 타겟과 독립변수 분리
y_train = dt_train['target']
X_train = dt_train.drop(['target'], axis=1)
X_test = dt_test.drop(['target'], axis=1)

# 학습용과 검증용 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=2023)

# 모델 학습
model = RandomForestRegressor(n_estimators=100, criterion='squared_error', random_state=1, n_jobs=-1)
model.fit(X_train, y_train)
print("모델 매개변수:", model)

# 검증 데이터 예측 및 성능 평가
pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, pred))
print(f'RMSE 검증: {rmse}')

# 피처 중요도 시각화
importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x=importances, y=importances.index)
plt.title("Feature Importances")
plt.show()

# 모델 저장
with open('saved_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Permutation Importance 계산
perm_importance = permutation_importance(model, X_val, y_val, scoring='neg_mean_squared_error', n_repeats=3,
                                         random_state=42)
importances_perm = perm_importance.importances_mean
std_perm = perm_importance.importances_std
indices = np.argsort(importances_perm)[::-1]
feature_names = X_val.columns.tolist()

# 검증 데이터에 예측값과 실제값 추가
X_val = X_val.copy()
X_val['target'] = y_val
X_val['pred'] = pred
X_val['error'] = (X_val['target'] - X_val['pred']) ** 2

# 오류가 큰 데이터와 작은 데이터 추출
X_val_sort = X_val.sort_values(by='error', ascending=False)
error_top100 = X_val_sort.head(100).copy()
best_top100 = X_val_sort.tail(100).copy()

# 레이블 인코딩 복원
for col in remaining_categorical_cols:
    # 'NULL'이 레이블 인코딩에 포함되었으므로 inverse_transform 시 오류가 발생하지 않습니다.
    error_top100[col] = label_encoders[col].inverse_transform(error_top100[col])
    best_top100[col] = label_encoders[col].inverse_transform(best_top100[col])

# 오류가 큰 데이터와 작은 데이터 시각화
sns.boxplot(data=error_top100, x='target')
plt.title('예측 오류 상위 100개 데이터의 Target 분포')
plt.show()

sns.boxplot(data=best_top100, x='target', color='orange')
plt.title('예측 성능 우수 상위 100개 데이터의 Target 분포')
plt.show()

sns.histplot(data=error_top100, x='전용면적(㎡)', alpha=0.5, label='오류 상위 100')
sns.histplot(data=best_top100, x='전용면적(㎡)', color='orange', alpha=0.5, label='성능 우수 상위 100')
plt.title('전용면적(㎡) 분포 비교')
plt.legend()
plt.show()

# 테스트 데이터 예측
with open('saved_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 테스트 데이터 예측 시간 측정 및 예측
start_time = time.time()
real_test_pred = model.predict(X_test)
elapsed_time = time.time() - start_time
print(f"예측 경과 시간: {elapsed_time} 초")

# 예측값을 정수형으로 변환 및 저장
real_test_pred = np.round(real_test_pred).astype(int)
preds_df = pd.DataFrame(real_test_pred, columns=["target"])
preds_df.to_csv('output_final.csv', index=False)

print(real_test_pred)  # 예측값 출력
