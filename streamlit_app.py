import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Definindo uma função para aplicar estilo ao texto do resultado
def resultado_estilizado(texto, cor_fundo="blue", cor_texto="white"):
    return f'<div style="font-size: 24px; padding: 10px; color: {cor_texto}; background-color: {cor_fundo}; border-radius: 5px;">{texto}</div>'

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.01, epochs=100):
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Inicializa os pesos com valores aleatórios pequenos
        self.weights = np.random.randn(num_inputs + 1)  # +1 para o viés

    def predict(self, inputs):
        # Adiciona 1 para o viés
        inputs = np.insert(inputs, 0, 1)
        # Calcula a soma ponderada
        weighted_sum = np.dot(inputs, self.weights)
        # Aplica a função de ativação (no caso, função degrau)
        activation = 1 if weighted_sum > 0 else 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                # Atualiza os pesos com base no erro
                error = label - prediction
                self.weights += self.learning_rate * error * np.insert(inputs, 0, 1)

def main():
    st.title("Perceptron de Rosenblatt para Classificação de Doenças Cardíacas")

    # Carregamento dos dados
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
                       header=None)
    data.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                    'slope', 'ca', 'thal', 'target']

    # Pré-processamento dos dados
    data = data.replace('?', np.nan)
    data = data.dropna()
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Criação e treinamento do perceptron
    perceptron = Perceptron(num_inputs=X_train_scaled.shape[1])
    perceptron.train(X_train_scaled, y_train)

    # Interface do usuário para inserir novos dados
    st.header("Novos Dados do Paciente")
    st.markdown("Preencha os dados do paciente abaixo para prever o risco de doenças cardíacas:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Idade", min_value=20, max_value=100, value=50)
        sex = st.radio("Sexo", ["Masculino", "Feminino"])
        sex = 1 if sex == "Masculino" else 0
        cp = st.slider("Tipo de Dor no Peito (Angina)", min_value=0, max_value=3, value=1,
                               help="0: Angina típica, 1: Angina atípica, 2: Dor não anginosa, 3: Assintomático")
        trestbps = st.slider("Pressão Arterial em Repouso", min_value=80, max_value=200, value=120,
                                     help="Quantidade de colesterol total no sangue, que é a soma das frações HDL, LDL e VLDL")
        chol = st.slider("Colesterol Sérico em mg/dl", min_value=100, max_value=600, value=200)
        fbs = st.radio("Nível de Açúcar no Sangue em Jejum", ["< 120 mg/dl", "> 120 mg/dl"])
        fbs = 1 if fbs == "> 120 mg/dl" else 0

    with col2:
        restecg = st.slider("Resultados do Eletrocardiograma em Repouso", min_value=0, max_value=2, value=1)
        thalach = st.slider("Frequência Cardíaca Máxima Alcançada", min_value=60, max_value=220, value=150)
        exang = st.radio("Angina Induzida pelo Exercício", ["Sim", "Não"])
        exang = 1 if exang == "Sim" else 0
        oldpeak = st.slider("Depressão do Segmento ST Induzida pelo Exercício em Relação ao Repouso",
                                    min_value=0.0, max_value=10.0, value=2.0,
                                    help="Alteração no traçado de um eletrocardiograma (ECG)")
        slope = st.slider("Inclinação do Segmento ST do Pico do Exercício", min_value=0, max_value=2, value=1)
        ca = st.slider("Número de Vasos Principais Coloridos por Fluoroscopia", min_value=0, max_value=4, value=0,
                               help="Técnica de imagem para obter imagens em tempo real de estruturas internas de um indivíduo através do uso de um fluoroscópio")
    thal = st.slider("Tipo de Defeito Cardíaco", min_value=0, max_value=3, value=2,
                     help="0: Normal, 1: Defeito fixo, 2: Defeito reversível, 3: Valor indefinido")

    new_patient = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
    new_patient_scaled = scaler.transform(new_patient.reshape(1, -1))
    prediction = perceptron.predict(new_patient_scaled)

    # Exibindo o resultado estilizado com cor dinâmica
    if prediction == 1:
        st.markdown(resultado_estilizado("O paciente está em risco de doenças cardíacas.", cor_fundo="#ff6961"), unsafe_allow_html=True)
    else:
        st.markdown(resultado_estilizado("O paciente não está em risco de doenças cardíacas.", cor_fundo="#90ee90", cor_texto="#333333"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
