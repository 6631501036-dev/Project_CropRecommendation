# 🌱 Crop Recommendation System (Thailand Soil & Climate Enriched)

ระบบแนะนำชนิดพืชที่เหมาะสมต่อการเพาะปลูกครอบคลุมพืชเศรษฐกิจและพืชเกษตร 22 ชนิด โดยผสานชุดข้อมูลปัจจัยการเจริญเติบโตของพืชเข้ากับ **ข้อมูลวิเคราะห์คุณภาพดิน (`soil.xml`)** และ **ข้อมูลค่าปกติทางอุตุนิยมวิทยารายจังหวัดของประเทศไทย (`climate_normal.xml`)** เพื่อสร้างโมเดล Machine Learning ที่สามารถทำนายชนิดพืชพร้อมระดับความมั่นใจ (Confidence Score) ได้อย่างแม่นยำ

---

## 📋 สารบัญ (Table of Contents)

* [ภาพรวมและจุดเด่นของโปรเจกต์](#-ภาพรวมและจุดเด่นของโปรเจกต์)
* [โครงสร้างไฟล์ในโปรเจกต์](#-โครงสร้างไฟล์ในโปรเจกต์)
* [สถาปัตยกรรมและขั้นตอนการทำงาน (Data & ML Pipeline)](#-สถาปัตยกรรมและขั้นตอนการทำงาน-data--ml-pipeline)
* [จังหวัดที่รองรับในระบบ (19 จังหวัดตัวแทนทุกภูมิภาค)](#-จังหวัดที่รองรับในระบบ-19-จังหวัดตัวแทนทุกภูมิภาค)
* [ผลการเปรียบเทียบประสิทธิภาพโมเดล](#-ผลการเปรียบเทียบประสิทธิภาพโมเดล)
* [การติดตั้งและวิธีรันโปรเจกต์](#-การติดตั้งและวิธีรันโปรเจกต์)
* [ตัวอย่างการนำโมเดลไปใช้งาน (Inference)](#-ตัวอย่างการนำโมเดลไปใช้งาน-inference)

---

## 🎯 ภาพรวมและจุดเด่นของโปรเจกต์

1. **Real-World Environmental Matching (การจับคู่สภาพแวดล้อมจริงรายจังหวัด):**
   * แทนที่จะสุ่มชื่อจังหวัด ระบบใช้อัลกอริทึม **1-Nearest Neighbors (Euclidean Distance)** บนข้อมูลที่ปรับมาตรฐานแล้ว (`StandardScaler`) จากตัวแปรร่วม 6 ปัจจัย ได้แก่ ฟอสฟอรัส ($P$), โพแทสเซียม ($K$), อุณหภูมิ (`temperature`), ความชื้นสัมพัทธ์ (`humidity`), ค่าความเป็นกรด-ด่าง ($\text{pH}$) และปริมาณน้ำฝนเฉลี่ยรายเดือน (`rainfall`) เพื่อจับคู่ข้อมูลพืชเข้ากับจังหวัดในประเทศไทยที่มีสภาพดินและภูมิอากาศใกล้เคียงที่สุดจริง
2. **Robust Data Preprocessing (การจัดการข้อมูลดินและอากาศอย่างแม่นยำ):**
   * **Soil Data:** จัดการโครงสร้าง JSON ภายในไฟล์ `soil.xml` สรุปค่าตัวแทนรายจังหวัดด้วยค่ามัธยฐาน (`median`) เพื่อลดผลกระทบจากค่าผิดปกติ (Outliers) และเติมค่าที่ขาดหายด้วยค่ามัธยฐานของแต่ละพารามิเตอร์
   * **Climate Data:** ดึงข้อมูลจากสถานีตรวจวัดหลักของแต่ละจังหวัดใน `climate_normal.xml` และคำนวณปริมาณน้ำฝนเป็นค่าเฉลี่ยรายเดือน (`np.mean`) เพื่อให้มีสเกลและหน่วยสอดคล้องกับชุดข้อมูลหลัก
3. **Production-Ready ML Pipeline:**
   * โมเดล **Random Forest Classifier** ทำความแม่นยำบนชุดทดสอบ (Test Set) ได้สูงถึง $99.55\%$ และถูกบันทึกพร้อมตัวแปลงข้อมูล (`StandardScaler`, `LabelEncoder`, และรายชื่อฟีเจอร์ทั้งหมด) ไว้ในไฟล์ `enriched_crop_pipeline.pkl` พร้อมนำไปเชื่อมต่อกับ Web Application หรือ API ได้ทันที

---

## 📁 โครงสร้างไฟล์ในโปรเจกต์

```text
Project_CropRecommendation/
├── Crop_recommendation/                    # โฟลเดอร์โปรเจกต์ตั้งต้น (Baseline Reference)
├── data/                                   # โฟลเดอร์เก็บชุดข้อมูลดิบและข้อมูลที่ประมวลผลแล้ว
│   ├── Crop_recommendation.csv             # ชุดข้อมูลดิบตั้งต้นก่อนทำความสะอาด
│   ├── crop_recommendation_cleaned.csv     # ข้อมูลตั้งต้นปัจจัยดินและอากาศของพืช 22 ชนิด (2,200 แถว)
│   ├── soil.xml                            # ข้อมูลผลวิเคราะห์คุณภาพดินรายจังหวัด (โครงสร้าง JSON)
│   ├── climate_normal.xml                  # ข้อมูลค่าปกติทางอุตุนิยมวิทยารายสถานี (โครงสร้าง XML)
│   ├── province_soil_climate_merged.csv    # [Generated] โปรไฟล์ดินและสภาพอากาศรวมรายจังหวัด (19 จังหวัด)
│   ├── enriched_crop_dataset.csv           # [Generated] ชุดข้อมูลพืชที่จับคู่คอลัมน์ location เรียบร้อยแล้ว
│   ├── enriched_crop_dataset_cleaned.csv   # [Generated] ชุดข้อมูลที่ผ่านการตรวจสอบและทำความสะอาดแล้ว
│   └── crop_recommendation_pipeline.pkl    # ไฟล์โมเดล Baseline (ก่อนผสานข้อมูลจังหวัด)
│
├── notebooks/                              # โฟลเดอร์เก็บ Jupyter Notebooks สำหรับ Data & ML Pipeline
│   ├── 00_prepairData.ipynb                # Step 1: รวมข้อมูล Soil + Climate และจับคู่จังหวัดด้วย Nearest Neighbors
│   ├── 01_data_cleaning_and_eda.ipynb      # Step 2: ตรวจสอบคุณภาพข้อมูล วิเคราะห์สถิติเชิงพรรณนา (EDA) และสร้างกราฟ
│   └── 02_model_training_and_export.ipynb  # Step 3: เทรนเปรียบเทียบ 5 โมเดล ประเมินผล และ Export Pipeline
│
├── enriched_crop_pipeline.pkl              # [Generated] ไฟล์โมเดลหลัก (Random Forest + Artifacts) ที่ Root Directory
└── README.md                               # เอกสารอธิบายโปรเจกต์ฉบับสมบูรณ์
```

---

## ⚙️ สถาปัตยกรรมและขั้นตอนการทำงาน (Data & ML Pipeline)

การทำงานทั้งหมดถูกแบ่งออกเป็น 3 ขั้นตอนหลักภายในโฟลเดอร์ `notebooks/` โดยอ้างอิงเส้นทางไฟล์ข้อมูลแบบ Relative Path (`../data/...`):

### Step 1: Data Preparation & Province Matching (`notebooks/00_prepairData.ipynb`)
* **Soil Parsing:** อ่านข้อมูลจาก `../data/soil.xml` สกัดค่าพารามิเตอร์ $P$, $K$, $\text{pH}$, $\text{OM}$, และ $\text{EC}$ แปลงชื่อจังหวัดภาษาไทยเป็นชื่อภาษาอังกฤษมาตรฐาน และสรุปค่าตัวแทนรายจังหวัดด้วย `median`
* **Climate Parsing:** อ่านค่าอุณหภูมิเฉลี่ย ความชื้นสัมพัทธ์เฉลี่ย และปริมาณน้ำฝนเฉลี่ยรายเดือนจาก `../data/climate_normal.xml` โดยให้ความสำคัญกับสถานีตรวจอากาศหลักของจังหวัดเป็นอันดับแรก
* **Profile Merging & Nearest Neighbor Matching:** รวมข้อมูลทั้งสองส่วนเข้าด้วยกันและเติมค่าว่างด้วยค่ามัธยฐาน บันทึกเป็น `../data/province_soil_climate_merged.csv` จากนั้นคำนวณระยะห่างแบบ Euclidean Distance หลังทำ Standardization เพื่อกำหนดคอลัมน์ `location` ให้กับข้อมูลทั้ง $2,200$ แถว และส่งออกเป็น `../data/enriched_crop_dataset.csv`

### Step 2: Data Cleaning & Exploratory Data Analysis (`notebooks/01_data_cleaning_and_eda.ipynb`)
* โหลดข้อมูลจาก `../data/enriched_crop_dataset.csv` เพื่อตรวจสอบค่าว่าง (Missing Values), ข้อมูลซ้ำซ้อน (Duplicates), ชนิดข้อมูล และความสมดุลของคลาสพืชทั้ง $22$ ชนิด
* สร้างกราฟวิเคราะห์การกระจายตัวของธาตุอาหารหลัก ($N, P, K$), ปัจจัยสภาพแวดล้อม (`temperature`, `humidity`, `ph`, `rainfall`), ความสัมพันธ์ระหว่างตัวแปร (Correlation Heatmap) และสัดส่วนจำนวนข้อมูลที่จับคู่ได้ในแต่ละจังหวัด ก่อนบันทึกเป็น `../data/enriched_crop_dataset_cleaned.csv`

### Step 3: Model Training, Evaluation & Export (`notebooks/02_model_training_and_export.ipynb`)
* โหลดข้อมูลจาก `../data/enriched_crop_dataset_cleaned.csv` แปลงตัวแปรกลุ่ม `location` ด้วย One-Hot Encoding (`pd.get_dummies`) และแปลงเป้าหมาย `label` ด้วย `LabelEncoder`
* แบ่งข้อมูล Train / Test ในสัดส่วน $80:20$ และปรับสเกลตัวแปรด้วย `StandardScaler`
* เทรนและเปรียบเทียบอัลกอริทึม 5 โมเดลด้วย 5-Fold Cross-Validation บันทึกโมเดลที่มีประสิทธิภาพสูงสุดลงใน `../enriched_crop_pipeline.pkl` และทดสอบทำนายผลจริงบนชุดข้อมูลตัวอย่าง (Verification Test)

---

## 🗺️ จังหวัดที่รองรับในระบบ (19 จังหวัดตัวแทนทุกภูมิภาค)

ข้อมูลอ้างอิงจากจังหวัดที่มีผลตรวจวิเคราะห์คุณภาพดินจริงใน `data/soil.xml` และจับคู่กับสถานีตรวจอากาศใน `data/climate_normal.xml` ครอบคลุมครบทุกภูมิภาคของประเทศไทย:

| ภูมิภาค | รายชื่อจังหวัด (`location` Key) |
| :--- | :--- |
| **ภาคเหนือ (10 จังหวัด)** | `CHIANG MAI`, `CHIANG RAI`, `MAE HONG SON`, `PHAYAO`, `PHRAE`, `TAK`, `KAMPHAENG PHET`, `SUKHOTHAI`, `NAKHON SAWAN`, `UTHAI THANI` |
| **ภาคตะวันออกเฉียงเหนือ (4 จังหวัด)** | `NAKHON RATCHASIMA`, `UDON THANI`, `UBON RATCHATHANI`, `ROI ET` |
| **ภาคกลางและตะวันตก (3 จังหวัด)** | `PATHUM THANI`, `RATCHABURI`, `PHETCHABURI` |
| **ภาคใต้ (2 จังหวัด)** | `SONGKHLA`, `TRANG` |

> **หมายเหตุสำหรับการนำไปสร้างแอปพลิเคชัน:** หากต้องการให้ผู้ใช้งานเลือกจังหวัดได้ครบทั้ง 77 จังหวัด สามารถสร้าง Mapping Dictionary ในฝั่ง Frontend/Backend เพื่อจับคู่จังหวัดที่อยู่นอกรายการเข้ากับจังหวัดตัวแทนในภูมิภาคใกล้เคียงที่มีสภาพภูมิอากาศคล้ายคลึงกันก่อนส่งค่าเข้าโมเดล

---

## 📊 ผลการเปรียบเทียบประสิทธิภาพโมเดล

จากการประเมินผลเปรียบเทียบ 5 อัลกอริทึมบนชุดข้อมูล `data/enriched_crop_dataset_cleaned.csv`:

| อันดับ | Model | CV Mean Accuracy ($\%$) | Test Accuracy ($\%$) | สถานะ |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Random Forest** | **$99.09\%$** | **$99.55\%$** | **Selected & Exported** |
| 2 | XGBoost | $98.92\%$ | $99.09\%$ | - |
| 3 | Decision Tree | $98.52\%$ | $98.41\%$ | - |
| 4 | K-Nearest Neighbors (KNN) | $93.81\%$ | $94.55\%$ | - |
| 5 | Gaussian Naive Bayes | $78.98\%$ | $77.73\%$ | - |

* จากการทำ **Verification Test** บนข้อมูลทดสอบตัวอย่าง 15 รายการแรก โมเดล **Random Forest** สามารถทำนายชนิดพืชได้ถูกต้องครบทั้ง 15 รายการ ($100\%$ Match) โดยมีค่าความมั่นใจ (Confidence Score) เฉลี่ยอยู่ในช่วง $75.0\% - 100.0\%$

---

## 🚀 การติดตั้งและวิธีรันโปรเจกต์

1. **ติดตั้งไลบรารีที่จำเป็น:**
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib jupyter
   ```

2. **รัน Jupyter Notebook ภายในโฟลเดอร์ `notebooks/` ตามลำดับ:**
   * **ขั้นตอนที่ 1:** เปิดและรัน `notebooks/00_prepairData.ipynb` เพื่อสร้างไฟล์ `data/province_soil_climate_merged.csv` และ `data/enriched_crop_dataset.csv`
   * **ขั้นตอนที่ 2:** เปิดและรัน `notebooks/01_data_cleaning_and_eda.ipynb` เพื่อสร้างไฟล์ `data/enriched_crop_dataset_cleaned.csv`
   * **ขั้นตอนที่ 3:** เปิดและรัน `notebooks/02_model_training_and_export.ipynb` เพื่อเทรนโมเดลและสร้างไฟล์ `enriched_crop_pipeline.pkl` ที่ Root Directory

---

## 💻 ตัวอย่างการนำโมเดลไปใช้งาน (Inference)

สามารถโหลดไฟล์ `enriched_crop_pipeline.pkl` จาก Root Directory ไปใช้งานร่วมกับ Web Framework (เช่น Streamlit, FastAPI, Flask) ได้ดังนี้:

```python
import joblib
import numpy as np
import pandas as pd

# 1. โหลด Pipeline และ Artifacts จาก Root Directory
pipeline = joblib.load('enriched_crop_pipeline.pkl')
model = pipeline['model']
scaler = pipeline['scaler']
le = pipeline['label_encoder']
feature_names = pipeline['feature_names']

# 2. เตรียมข้อมูล Input ใหม่ (ค่าธาตุอาหารในดิน สภาพอากาศ และจังหวัด)
sample_input = pd.DataFrame([{
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 26.5,
    'humidity': 80.0,
    'ph': 6.5,
    'rainfall': 180.0,
    'location': 'CHIANG MAI'
}])

# 3. ทำ One-Hot Encoding และจัดเรียงคอลัมน์ให้ตรงกับตอน Train
input_encoded = pd.get_dummies(sample_input).reindex(columns=feature_names, fill_value=0)
input_scaled = scaler.transform(input_encoded)

# 4. ทำนายชนิดพืชและคำนวณระดับความมั่นใจ (Confidence Score)
pred_idx = model.predict(input_scaled)[0]
pred_proba = np.max(model.predict_proba(input_scaled)) * 100
recommended_crop = le.inverse_transform([pred_idx])[0]

print(f"พืชที่แนะนำ: {recommended_crop} (ความมั่นใจ: {pred_proba:.2f}%)")
```