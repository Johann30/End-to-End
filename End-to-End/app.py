# Importa las bibliotecas necesarias
import streamlit as st  
import pandas as pd    
import numpy as np     
from sklearn.compose import ColumnTransformer  
from sklearn.pipeline import Pipeline          
from sklearn.impute import SimpleImputer        
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder  
from sklearn.base import BaseEstimator, TransformerMixin  
import joblib      

# Función para predecir el valor de una casa
def predicts(data):
    # Carga el modelo preentrenado desde el archivo "my_model.pkl"
    model = joblib.load("my_model.pkl")
    
    # Obtiene la pipeline completa para preparar los datos
    full_pipeline = fetch_pipeline()
    
    # Transforma los datos de entrada según la pipeline
    data_prepared = full_pipeline.transform(data)
    
    # Realiza la predicción con el modelo
    return model.predict(data_prepared)

# Función para obtener la pipeline de preparación de datos
def fetch_pipeline():
    # Define una pipeline numérica para transformaciones en datos numéricos
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),  # Imputa valores faltantes con la mediana
        ('attribs_adder', CombinedAttributesAdder()),   # Agrega atributos personalizados
        ('std_scaler', StandardScaler())                # Estandariza los datos
    ])

    # Lee el conjunto de datos de viviendas desde un archivo CSV
    housing = pd.read_csv("datasets/housing/housing.csv")
    
    # Elimina la columna "median_house_value" del conjunto de datos
    housing = housing.drop("median_house_value", axis=1)
    
    # Divide el conjunto de datos en datos numéricos y categóricos
    housing_num = housing.drop("ocean_proximity", axis=1)
    
    # Define las columnas numéricas y categóricas
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    # Crea una ColumnTransformer para aplicar transformaciones a diferentes columnas
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),    # Aplica la pipeline numérica a las columnas numéricas
        ("cat", OneHotEncoder(), cat_attribs)  # Codifica las columnas categóricas
    ])
    
    # Ajusta la pipeline completa al conjunto de datos de viviendas
    full_pipeline.fit(housing)

    return full_pipeline

# Índices de las columnas específicas en los datos
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

# Clase para agregar atributos personalizados
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# Configura la interfaz de usuario de Streamlit
st.title("Prediction Model")

# Entrada de usuario para diferentes características de viviendas
longitude = st.number_input("Longitude", max_value=0.0)
latitude = st.number_input("Latitude", min_value=1.0)
age = st.number_input("Housing Median Age", min_value=1.0)
total_rooms = st.number_input("Total rooms", min_value=1.0)
total_bedrooms = st.number_input("Total bedrooms", min_value=1.0)
population = st.number_input("Population", min_value=1.0)
households = st.number_input("Households", min_value=1.0)
income = st.number_input("Median income", min_value=1.0)

# Selección de una categoría para la proximidad al océano
ocean_proximity = st.selectbox("Ocean Proximity: ",['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

# Botón para realizar la predicción del valor de la casa
if st.button("Predict house value"):
    # Mapea la categoría de proximidad al océano a un valor numérico
    ocean = 0 if ocean_proximity == '<1H OCEAN' else 1 if ocean_proximity == 'INLAND' else 2 if ocean_proximity == 'ISLAND' else 3 if ocean_proximity == 'NEAR BAY' else 4
    
    # Crea un DataFrame con los valores de entrada del usuario
    data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [income],
        'ocean_proximity': [ocean_proximity]
    })
    
    # Realiza la predicción y muestra el resultado en la interfaz
    result = predicts(data)
    st.text(result[0])

st.write("<p style='text-align: center;'>By Johann Lozano Enriquez  a338834</p>", unsafe_allow_html=True)
