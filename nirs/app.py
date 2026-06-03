import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from catboost import CatBoostClassifier
from sklearn.metrics import f1_score


@st.cache_resource
def load_data():
    with open("assets/age_scaler.pkl", "rb") as f:
        age_scaler = pickle.load(f)
    with open("assets/hours_scaler.pkl", "rb") as f:
        hours_scaler = pickle.load(f)
    with open("assets/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    with open("assets/X_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open("assets/y_train.pkl", "rb") as f:
        y_train = pickle.load(f)
    with open("assets/X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open("assets/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)
    return age_scaler, hours_scaler, feature_cols, X_train, y_train, X_test, y_test


(
    age_scaler,
    hours_scaler,
    feature_cols,
    X_train,
    y_train,
    X_test,
    y_test,
) = load_data()


@st.cache_resource(
    show_spinner="Обучение модели с выбранными гиперпараметрами...",
    show_time=True,
)
def train_model(depth: int, iterations: int, learning_rate: float):
    model = CatBoostClassifier(
        depth=depth,
        iterations=iterations,
        learning_rate=learning_rate,
        eval_metric="F1",
        verbose=False,
        random_seed=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    return model, f1


st.sidebar.header("Настройка CatBoost")

depth = st.sidebar.slider("Глубина деревьев (depth)", 1, 16, 10)
iterations = st.sidebar.slider("Число итераций (iterations)", 100, 2000, 100, step=50)
learning_rate = st.sidebar.slider(
    "Скорость обучения (learning_rate)", 0.001, 0.3, 0.01, step=0.001
)

st.sidebar.caption("При изменении параметров модель переобучается.")

model, f1_val = train_model(depth, iterations, learning_rate)
st.sidebar.markdown(f"**F1 на тесте:** `{f1_val:.3f}`")


EDUCATION_OPTIONS = [
    ("Preschool", "Дошкольное"),
    ("1st-4th", "1–4 класс"),
    ("5th-6th", "5–6 класс"),
    ("7th-8th", "7–8 класс"),
    ("9th", "9 класс"),
    ("10th", "10 класс"),
    ("11th", "11 класс"),
    ("12th", "12 класс"),
    ("HS-grad", "Среднее (аттестат)"),
    ("Some-college", "Неоконченное высшее"),
    ("Assoc-voc", "Среднее проф. (техническое)"),
    ("Assoc-acdm", "Среднее проф. (академическое)"),
    ("Bachelors", "Бакалавр"),
    ("Masters", "Магистр"),
    ("Prof-school", "Профессиональная школа"),
    ("Doctorate", "Докторантура"),
]

OCCUPATION_OPTIONS = [
    ("?", "Неизвестно"),
    ("Adm-clerical", "Адм. персонал"),
    ("Armed-Forces", "Вооружённые силы"),
    ("Craft-repair", "Ремесло и ремонт"),
    ("Exec-managerial", "Руководитель / менеджер"),
    ("Farming-fishing", "Сельское хозяйство / рыболовство"),
    ("Handlers-cleaners", "Грузчики / уборщики"),
    ("Machine-op-inspct", "Оператор машин / контролёр"),
    ("Other-service", "Прочие услуги"),
    ("Priv-house-serv", "Домашний персонал"),
    ("Prof-specialty", "Проф. специальность"),
    ("Protective-serv", "Охрана / силовые структуры"),
    ("Sales", "Продажи"),
    ("Tech-support", "Тех. поддержка"),
    ("Transport-moving", "Транспорт / перевозки"),
]

EDUCATION_CATS = [v for v, _ in EDUCATION_OPTIONS]
OCCUPATION_CATS = [v for v, _ in OCCUPATION_OPTIONS]
edu_labels = [label for _, label in EDUCATION_OPTIONS]
occ_labels = [label for _, label in OCCUPATION_OPTIONS]


def preprocess(age, hours, sex, race, is_married, education, occupation):
    age_s = float(age_scaler.transform([[age]])[0][0])
    hours_s = float(hours_scaler.transform([[hours]])[0][0])

    row = {
        "age": age_s,
        "race": race,
        "sex": sex,
        "hours.per.week": hours_s,
        "is_married": is_married,
    }

    for cat in EDUCATION_CATS:
        row[f"education_{cat}"] = 1 if education == cat else 0
    for cat in OCCUPATION_CATS:
        row[f"occupation_{cat}"] = 1 if occupation == cat else 0

    df_row = pd.DataFrame([row])
    df_row = df_row[feature_cols]
    return df_row


def predict(input_df):
    proba = model.predict_proba(input_df)[0]
    pred = int(proba[1] >= 0.5)
    return pred, proba


st.title("Прогноз дохода >$50K в год")
st.subheader("Введите характеристики:")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Возраст", 17, 90, 35)
    hours = st.slider("Часов в неделю", 1, 99, 40)
    is_married = st.selectbox(
        "В браке", [0, 1], format_func=lambda x: "Да" if x else "Нет"
    )

with col2:
    sex = st.selectbox(
        "Пол", [0, 1], format_func=lambda x: "Женский" if x else "Мужской"
    )
    race = st.selectbox(
        "Цвет кожи", [0, 1], format_func=lambda x: "Чёрный" if x else "Белый"
    )

with col3:
    edu_idx = st.selectbox(
        "Образование",
        range(len(EDUCATION_OPTIONS)),
        format_func=lambda i: edu_labels[i],
        index=12,
    )
    education = EDUCATION_CATS[edu_idx]

    occ_idx = st.selectbox(
        "Занятие",
        range(len(OCCUPATION_OPTIONS)),
        format_func=lambda i: occ_labels[i],
        index=10,
    )
    occupation = OCCUPATION_CATS[occ_idx]

input_df = preprocess(age, hours, sex, race, is_married, education, occupation)
pred, proba = predict(input_df)

st.markdown("---")
res_col, chart_col = st.columns(2)

with res_col:
    st.subheader("Результат")
    if pred == 1:
        st.success("Доход **>$50K**")
    else:
        st.info("Доход **≤$50K**")
    st.metric("Вероятность >$50K", f"{proba[1]:.1%}")
    st.metric("Вероятность ≤$50K", f"{proba[0]:.1%}")

with chart_col:
    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(["≤$50K", ">$50K"], proba, color=["#e74c3c", "#2ecc71"])
    ax.bar_label(bars, labels=[f"{p:.1%}" for p in proba], padding=3)
    ax.set_ylabel("Вероятность")
    ax.set_ylim(0, 1.1)
    ax.set_title("Вероятности классов")
    st.pyplot(fig)
