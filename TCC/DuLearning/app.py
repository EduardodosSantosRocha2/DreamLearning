from flask import Flask, request, jsonify, render_template

# Importando a biblioteca do Random Forest(Floresta aleatoria, pois é um conjunto de arvores)
from sklearn.ensemble import RandomForestClassifier
# Importando a biblioteca da MÁQUINAS DE VETORES DE SUPORTE (SVM)
from sklearn.svm import SVC
# Importando a biblioteca da REGRESSÃO LOGÍSTICA(Apesar do nome regressaão que lembre previsão, mas nesse caso ele é um algoritmo de classificação)
from sklearn.linear_model import LogisticRegression
# Importando a biblioteca da APRENDIZAGEM BASEADA EM INSTÂNCIAS (KNN)
from sklearn.neighbors import KNeighborsClassifier
# Importando a biblioteca da ÁRVORE DE DECISÃO
from sklearn.tree import DecisionTreeClassifier
# Importando a biblioteca do XGBOOST (pip install xgboost) ou no colab(pip install xgboost)
from xgboost import XGBClassifier
# Importando a biblioteca do LIGHTGBM (!pip install lightgbm) ou no colab(pip install lightgbm)
import lightgbm as lgbm
# Importando a biblioteca do CATBOOST (!pip install catboost) ou no colab(pip install catboost)
from catboost import CatBoostClassifier

# Importando a biblioteca LABELENCOLDER para TRANSFORMAÇÃO DE VARIAVEIS CATEGORICAS EM NUMERICAS
from sklearn.preprocessing import LabelEncoder
# Importando a biblioteca para avaliação do algoritimo
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # Importando a biblioteca para avaliação do algoritimo

# treinamento
from sklearn.model_selection import train_test_split

import pandas as pd
import io
import math
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px

# Regressão
 
# Análise da Normalidade
import scipy.stats as stats

#Teste Lilliefors (Kolmogorov_Sminorv) Ho = distribuição normal : p > 0.05 Ha = distribuição != normal : p <= 0.05 (pip install statsmodels)
import statsmodels
from statsmodels.stats.diagnostic import lilliefors

# Importando a biblioteca da regressão linear simples, mutipla e polinomial
from sklearn.linear_model import LinearRegression
#Importando a biblioteca polinomial
from sklearn.preprocessing import PolynomialFeatures
# Importando a biblioteca da SVM 
from sklearn.svm import SVR
# Importando a biblioteca da Arvore de decisão
from sklearn.tree import DecisionTreeRegressor
# Importando a biblioteca do Random Forest
from sklearn.ensemble import RandomForestRegressor
# Importando a biblioteca do XGBOOST
from xgboost import XGBRegressor
# Importando a biblioteca do lightgbm(import lightgbm as lgb)
    
# Importando a biblioteca do lightgbm
from catboost.core import CatBoostRegressor

#Padronização de escala
from sklearn.preprocessing import StandardScaler
#Metricas de desempenho
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Validação cruzada
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#Biblioteca Eclat Regras de associaçao(pip install pyECLAT), Apriori (pip install mlxtend)
from pyECLAT import ECLAT
from mlxtend.frequent_patterns import apriori, association_rules



import warnings
warnings.filterwarnings("ignore")

import google.generativeai as geneai

from classifier_data import classifierData



# Gemini 
from flask import Flask, request, jsonify, render_template
import datetime
import markdown
import google.generativeai as geneai

GOOGLE_GEMINI_API_KEY = "AIzaSyBkSzcTf_FBw73YXNZ8X4B1SRE1Dp9E6nI"
geneai.configure(api_key=GOOGLE_GEMINI_API_KEY)
model = geneai.GenerativeModel("gemini-1.5-flash-8b")

app = Flask(__name__)



@app.route("/Home")
def home():
    return render_template("home.html")

@app.route("/")
def login():
    return render_template("login.html")

# Rota para o frontend
@app.route("/Classificacao")
def index():
    return render_template("classifiers.html")

# Rota para o teste de normalidade e correlação
@app.route("/Normalidadeecorrelação")
def typedata():
    return render_template("typedata.html")


@app.route("/Regressao")
def regression():
    return render_template("regression.html")

@app.route("/Analisegrafica")
def graphicAnalysis():
    return render_template('graphicAnalysis.html')

@app.route("/RegrasAssociacao")
def associationRules():
    return render_template('associationRules.html')

@app.route("/VideoAulas")
def videoAulas():
    return render_template("videoAulas.html")

@app.route("/videoAulaRandomForest")
def videoAulasRandomForest():
    return render_template("videoAulaRandomForest.html")

@app.route("/videoAulaNaiveBayes")
def videoAulaNaiveBayes():
    return render_template("videoAulaNaiveBayes.html")

@app.route("/videoAulaSvm")
def videoAulaSvm():
    return render_template("videoAulaSvm.html")


@app.route("/videoAulaRegressaoLogistica")
def videoAulaRegressaoLogistica():
    return render_template("videoAulaRegressaoLogistica.html")


@app.route("/videoAulaKnn")
def videoAulaKnn():
    return render_template("videoAulaKnn.html")


@app.route("/videoAulaArvoreDecisao")
def videoAulaArvoreDecisao():
    return render_template("videoAulaArvoreDecisao.html")


@app.route("/videoAulaXGboost")
def videoAulaXGboost():
    return render_template("videoAulaXGboost.html")


@app.route("/videoAulaLightGBM")
def videoAulaLightGBM():
    return render_template("videoAulaLightGBM.html")


@app.route("/videoAulaCatBoost")
def videoAulaCatBoost():
    return render_template("videoAulaCatBoost.html")


@app.route("/videoAulaRegressaoLinear")
def videoAulaRegressaoLinear():
    return render_template("videoAulaRegressaoLinear.html")


@app.route("/videoAulaRegressaoLinearMultipla")
def videoAulaRegressaoLinearMultipla():
    return render_template("videoAulaRegressaoLinearMultipla.html")



@app.route("/videoAulaRegressaoPolinomial")
def videoAulaRegressaoPolinomial():
    return render_template("videoAulaRegressaoPolinomial.html")

@app.route("/videoAulaRegrasdeAssociacao")
def videoAulaRegrasdeAssociacao():
    return render_template("videoAulaRegrasdeAssociacao.html")

@app.route("/Historico")
def historic():
    return render_template("historico.html")


@app.route("/predict", methods=["POST"])
def teste():
    # Recebe os dados enviados pelo formulário
    classifier_type = request.form["classifier"]
    print(classifier_type)
    csv_file = request.files["csv_file"]      
    separator = request.form.get('separator')
    # Lê o arquivo CSV

    csv_data = pd.read_csv(io.BytesIO(csv_file.read()),sep = separator, encoding = 'utf-8') 
    csv_tranform = csv_data.copy()

    csv_deploy = ""
    deployBoolean = request.form.get('deployBoolean')
    if deployBoolean == "true":
        csv_file_deploy = request.files["csv_deploy"]
        csv_deploy = pd.read_csv(io.BytesIO(csv_file_deploy.read()), sep= separator, encoding='utf-8', header=None)
        print(f"-----------------------------------------------------")
        csv_deploy = csv_deploy.to_numpy().tolist()
        print(csv_deploy)
        print(f"-----------------------------------------------------")

    
    
    classifier_data = classifierData()
    
    csv_data_encolder, meu_dicionario, meu_dicionarioencoder = classifier_data.encode_columns(csv_data)
    x_treino, x_teste, y_treino, y_teste = classifier_data.train_test_split(csv_data_encolder)
    resultado= classifier_data.train(classifier_type,x_treino, x_teste, y_treino, y_teste, separator, deployBoolean, csv_tranform, csv_data_encolder, csv_deploy, meu_dicionario, meu_dicionarioencoder)
    return resultado


# Rota para fazer testes de normalidade e correlação para regressão linear simples
@app.route("/typedatatest", methods=["POST"])
def typedatatest():
    csv_file = request.files["csv_file"]
    separator = request.form.get('separator')
    
    # Lê o arquivo CSV
    csv_data = pd.read_csv(io.BytesIO(csv_file.read()), sep=separator, encoding='utf-8')
    
    print(csv_data)
    
    # Aplica LabelEncoder se necessário
    encoder = LabelEncoder()
    for col in csv_data.columns:
        if csv_data[col].dtype == 'object':
            csv_data[col] = encoder.fit_transform(csv_data[col])
    
    print(csv_data)
    
    # Função para verificar normalidade dos dados
    def TesteLilliefors():
        continuesNormality = True
        for col in csv_data.columns:
            estatistica, p = statsmodels.stats.diagnostic.lilliefors(csv_data[col], dist='norm')
            if p <= 0.05:
                continuesNormality = False
            print('Estatistica de teste: {}'.format(estatistica))
            print('p-valor: {}'.format(p))
        return continuesNormality
    
    continuesNormality = TesteLilliefors()
    
    # Função para calcular correlação linear
    def CorrelaçãoLinear():
        if continuesNormality:
            correlation = csv_data.corr(method='pearson')
        else:
            correlation = csv_data.corr(method='spearman')
        
        # Converter para JSON com orient='columns' para manter os cabeçalhos das colunas
        json_correlation = correlation.to_json(orient='columns')
        print(json_correlation)

        return json_correlation    

    json_correlation = CorrelaçãoLinear()    
    
    # Retorna os valores dos testes
    return jsonify({"linearCorrelation": json_correlation, "continuesNormality": continuesNormality})





# Rota para fazer regressão
@app.route("/regressionPost", methods=["POST"])
def regressionPost():

    global parameters
    parameters = []

    global categoricas
    categoricas = []

    # Recebe os dados enviados pelo formulário
    classifier_type = request.form["regression"]
    print(classifier_type)
    csv_file = request.files["csv_file"] 
    separator = request.form.get('separator')
    posicao = request.form.get('posicao')
    print(f"A posicao eh {posicao}")
    deployBoolean = request.form.get('deployBoolean')
    posicao = request.form.get('posicao')
    crossVal = request.form.get('crossVal')
    print("----------------------------------")
    print(crossVal)
    print("----------------------------------")
    code = ""

   
    

    # Lê o arquivo CSV
    csv_data = pd.read_csv(io.BytesIO(csv_file.read()),sep = separator, encoding = 'utf-8')
    csv_tranform = csv_data.copy()
    print(csv_data)


    if deployBoolean == "true":
        csv_file_deploy = request.files["csv_deploy"]
        csv_deploy = pd.read_csv(io.BytesIO(csv_file_deploy.read()), sep= separator, encoding='utf-8', header=None)
        print(f"-----------------------------------------------------")
        csv_deploy = csv_deploy.to_numpy().tolist()
        print(csv_deploy)
        print(f"-----------------------------------------------------")
    
    
    
    #Vetores necessarios para diferentes tipos de regressão e diferentes x e y
    name_Regression_fit = ["SIMPLE LINEAR", "MULTIPLE LINEAR", "DECISION TREE", "RANDOM FOREST","XGBOOST","LIGHT GBM", "CATBOOST"]
    name_Regression_poly = ["POLYNOMIAL"]
    name_Regression_SVR = ["SUPPORT VECTORS(SVR)"]



    #Aplica o labelencolder
    meu_dicionario = {}
    meu_dicionarioencoder = {}

    dicionarioParametros = {}

    # Inicialize o encoder
    encoder = LabelEncoder()

    # Itere sobre todas as colunas do DataFrame
    for coluna in csv_data.columns:
        # Verifique se a coluna é do tipo 'object' e não é numérica
        if csv_data[coluna].dtype == 'object' and not csv_data[coluna].apply(lambda x: isinstance(x, (int, float))).all():
            # Imprima os valores únicos antes e depois da codificação para essa coluna
            print(f'Coluna: {coluna}')
            print(f'Valores unicos antes da codificacao: {csv_data[coluna].unique()}')
            
            meu_dicionario[coluna] = csv_data[coluna].unique().tolist()
            
            # Ajuste o encoder aos dados e transforme a coluna
            csv_data[coluna] = encoder.fit_transform(csv_data[coluna])
            
            print(f'Valores únicos depois da codificacao: {csv_data[coluna].unique()}')
            meu_dicionarioencoder[coluna] = csv_data[coluna].unique().tolist()
            print('\n')

    print(f"Dicionario: {meu_dicionario}")
    print(f"Dicionario Encoder: {meu_dicionarioencoder}")
    print(csv_tranform)




    def parameters_forms():
        # Recebe as características do parametros
        parameters = []
        i = 1
        while True:
            parametro = request.form.get(f"parameters{i}")
            if parametro is None:
                break
            parameters.append(parametro)
            print(f"Parametro {i}: {parametro}")
            i += 1
        print(f"Olha eles: {parameters}")
        return parameters
    
  

    def train_fit(reg,x_teste, x_treino, y_teste, y_treino):    
        reg.fit(x_treino, y_treino)
        determinationCoefficientTraining = reg.score(x_treino, y_treino)
        determinationCoefficientTest  = reg.score(x_teste, y_teste)
        previsoes_teste = reg.predict(x_teste)
        absolute = mean_absolute_error(y_teste, previsoes_teste)
        MeanSquaredError = np.sqrt(mean_squared_error(y_teste, previsoes_teste))
        return determinationCoefficientTraining,determinationCoefficientTest,absolute,MeanSquaredError
    
    def train_poly(reg,x_teste, x_treino, y_teste, y_treino):    
        # Pré Processamento
        grau_polinomial =PolynomialFeatures(degree=2)
        x_poly = grau_polinomial.fit_transform(x_treino)      
        reg.fit(x_poly, y_treino)
        determinationCoefficientTraining = reg.score(x_poly, y_treino)       
        x_poly_teste = grau_polinomial.fit_transform(x_teste)
        polinomial_teste = LinearRegression()
        polinomial_teste.fit(x_poly_teste, y_teste)
        determinationCoefficientTest  = reg.score(x_poly_teste, y_teste)
        previsoes_teste = polinomial_teste.predict(x_poly_teste)
        absolute = mean_absolute_error(y_teste, previsoes_teste)
        MeanSquaredError = np.sqrt(mean_squared_error(y_teste, previsoes_teste)) 
        return determinationCoefficientTraining,determinationCoefficientTest,absolute,MeanSquaredError
    
    def train_SVR(reg,x_teste, x_treino, y_teste, y_treino):    
        # Pré Processamento escalonando
        x_scaler = StandardScaler()
        x_treino_scaler = x_scaler.fit_transform(x_treino)
        y_scaler = StandardScaler()
        y_treino_scaler = y_scaler.fit_transform(y_treino.to_numpy().reshape(-1,1))
        x_teste_scaler = x_scaler.transform(x_teste)
        y_teste_scaler = y_scaler.transform(y_teste.to_numpy().reshape(-1,1))  
        #treinando
        reg.fit(x_treino_scaler, y_treino_scaler.ravel())# .ravel() é para retornar matriz 1D
        #coeficientes de determinação
        determinationCoefficientTraining = reg.score(x_treino_scaler, y_treino_scaler)
        determinationCoefficientTest  = reg.score(x_teste_scaler, y_teste_scaler)
        #Predict x teste
        previsoes_teste = reg.predict(x_teste_scaler)
        #Revertendo a transformação
        y_teste_inverse = y_scaler.inverse_transform(y_teste_scaler)
        previsoes_inverse = y_scaler.inverse_transform(previsoes_teste.reshape(-1, 1))
        
        #Metricas
        absolute = mean_absolute_error(y_teste_inverse, previsoes_inverse)
        MeanSquaredError = np.sqrt(mean_squared_error(y_teste_inverse, previsoes_inverse))

        return determinationCoefficientTraining,determinationCoefficientTest,absolute,MeanSquaredError
    
    
    
    def some_columns():
        # Obtendo o índice da coluna x
        select_Independent_Variable = request.form["csv_headers"]
        indice_coluna = csv_data.columns.get_loc(select_Independent_Variable)
        # Separa as características e o alvo
        X = csv_data.iloc[:, indice_coluna:indice_coluna+1].values
        y =  csv_data.iloc[:, -1].values
        return X, y

    def all_columns():    
        # Separa as características e o alvo
        X = csv_data.iloc[:, :-1]
        y = csv_data.iloc[:, -1]
        print(X, y)
        return X, y
    
    def deployBooleanTrue(csv_data, csv_tranform, csv_deploy, meu_dicionario, meu_dicionarioencoder, reg):
        print(csv_data)
        print(csv_tranform)

        cont_coluns = 0
        deploy = csv_deploy
        print(f"Deploy tem tamanho: {len(deploy)}")

        for i in range(len(deploy)):
            num_features = len(deploy[i])  # Armazena o comprimento da lista de features
            cont_coluns  = 0
            for coluna in csv_tranform.columns:
                # Verifica se o tipo da coluna é 'object' e se todos os elementos não são números
                if csv_tranform[coluna].dtype == 'object' and not csv_tranform[coluna].apply(lambda x: isinstance(x, (int, float))).all():
                    # Verifica se o índice está dentro do alcance da lista de features
                    if cont_coluns < num_features:
                        # Codifica a feature usando o dicionário
                        deploy[i][cont_coluns] = meu_dicionarioencoder[coluna][meu_dicionario[coluna].index(deploy[i][cont_coluns])]
                cont_coluns += 1

        
        print(f"Printando o deploy : {deploy}")
        prediction = reg.predict(deploy)
        val = prediction.tolist()
        return val
    
    def crossValidation(reg,X,y):
        kfold = KFold(n_splits = 15, shuffle=True, random_state = 5)
        result = cross_val_score(reg,X, y,cv = kfold) 
        return result.mean()*100
 
    if(classifier_type == "SIMPLE LINEAR"):
        reg =  LinearRegression()
        X,y = some_columns()

        
    elif(classifier_type == "MULTIPLE LINEAR"):
        reg =  LinearRegression()
        X,y = all_columns()
        code+= f"""\n<span class="keyword">independentes</span><span> = dataframe.iloc[:, :-1]</span>\n<span class="keyword">dependentes</span><span> = dataframe.iloc[:, -1]</span>\n<span class="bib">from</span> sklearn.model_selection <span class="bib">import</span> train_test_split\n<span class="keyword">x_treino, x_teste, y_treino, y_teste =</span> <span class="function">train_test_split(independentes,dependentes, test_size = 0.3, random_state = 0)</span><span class="comment">#Base treino e teste</span>\n\n<span class="comment">#Importando biblioteca da regressão múltipla</span>\n<span><span class="bib">from </span>sklearn.linear_model <span class="bib">import </span>LinearRegression</span>\n<span><span class="keyword">regressao_multipla </span><span>= </span><span class="function">LinearRegression</span>()</span>\n<span><span class="keyword">regressao_multipla</span>.<span class="function">fit</span>(x_treino, y_treino)</span>\n\n<span class="comment">#Coeficiente de Determinação Treino</span>\n<span><span class="keyword">coeficiente_determinacao_treino = </span>regressao_multipla.<span class="function">score</span>(x_treino, y_treino)</span>\n<span><span class="function">print</span>(f"Coeficiente de determinação treino: {{coeficiente_determinacao_treino}}")</span>\n\n<span class="comment">#Coeficiente de Determinação Teste</span>\n<span><span class="keyword">coeficiente_determinacao_teste = </span>regressao_multipla.<span class="function">score</span>(x_teste, y_teste)</span>\n<span><span class="function">print</span>(f"Coeficiente de determinação teste: {{coeficiente_determinacao_teste}}")</span>\n\n<span class="comment">#Realiza previsões do teste</span>\n<span><span class="keyword">previsoes</span> = regressao_multipla.<span class="function">predict</span>(x_teste)</span>\n\n<span class="comment">#Biblioteca de métricas </span>\n<span><span class="bib">from </span>sklearn.metrics <span class="bib">import </span>mean_absolute_error, mean_squared_error</span>\n\n<span class="comment">#Erro médio Absoluto</span>\n<span><span class="keyword">erro_medio_absoluto</span> = <span class="function">mean_absolute_error</span>(y_teste, previsoes)</span>\n<span><span class="function">print</span>(f"Erro médio Absoluto: {{erro_medio_absoluto}}")</span>\n\n\n<span class="comment">#Erro quadrático médio</span>\n<span><span class="keyword">erro_quadratico_medio</span>= <span class="function">mean_squared_error</span>(y_teste, previsoes)\n</span><span><span class="function">print</span>(f"Erro quadrático médio: {{erro_quadratico_medio}}")</span>\n\n<span class="comment">#Raiz do erro quadrático médio</span>\n<span><span class="keyword">raiz_erro_quadratico_medio</span>= <span class="function">np.sqrt</span>(mean_squared_error(y_teste, previsoes))</span>\n</span><span><span class="function">print</span>(f"Raiz do erro quadrático médio: {{raiz_erro_quadratico_medio}}")</span>\n"""
    
    elif(classifier_type == "POLYNOMIAL"):
        reg =  LinearRegression()
        X,y = some_columns()
        # Obtendo o índice da coluna x
        select_Independent_Variable = request.form["csv_headers"]
        indice_coluna = csv_data.columns.get_loc(select_Independent_Variable)
        code+= f"""\n<span class="keyword">independentes</span><span> = dataframe.iloc[:, {indice_coluna}:{indice_coluna+1}]</span>\n<span class="keyword">dependentes</span><span> = dataframe.iloc[:, -1]</span>\n<span class="bib">from</span> sklearn.model_selection <span class="bib">import</span> train_test_split\n<span class="keyword">x_treino, x_teste, y_treino, y_teste =</span> <span class="function">train_test_split(independentes,dependentes, test_size = 0.3, random_state = 0)</span><span class="comment">#Base treino e teste</span>\n\n<span class="comment">#Importando biblioteca de pré processamento para transformação polinomial </span>\n<span><span class="bib">from </span>sklearn.preprocessing <span class="bib">import </span> PolynomialFeatures</span>\n\n<span class="comment">#Importando biblioteca da regressão múltipla</span>\n<span><span class="bib">from </span>sklearn.linear_model <span class="bib">import </span>LinearRegression</span>\n\n<span class="comment">#Pré Processamento</span>\n<span><span class="keyword">grau_polinomial </span>= <span class="function">PolynomialFeatures</span>(degree=2)</span>\n\n<span class="comment">#Transforma os dados de treino em um espaço polinomial</span>\n<span><span class="keyword">x_poly_treino </span> = grau_polinomial.<span class="function">fit_transform</span>(x_treino)</span>\n\n\n<span class="comment">#Cria um modelo de regressão linear para os dados de treino polinomiais</span>\n<span><span class="keyword">regressao_polinomial_treino</span> = <span class="function">LinearRegression</span>()</span>\n\n\n<span class="comment">#Treina o modelo de regressão polinomial com os dados de treino</span>\n<span><span class="keyword">regressao_polinomial_treino</span>.<span class="function">fit</span>(x_poly_treino, y_treino)</span>\n\n\n<span class="comment">#Transforma os dados de teste em um espaço polinomial</span>\n<span><span class="keyword">x_poly_teste </span> = grau_polinomial.<span class="function">fit_transform</span>(x_teste)</span>\n\n<span class="comment">#Cria um modelo de regressão linear para os dados de teste polinomiais</span>\n<span><span class="keyword">regressao_polinomial_teste</span> = <span class="function">LinearRegression</span>()</span>\n\n<span class="comment">#Treina o modelo de regressão polinomial com os dados de teste</span>\n<span><span class="keyword">regressao_polinomial_teste</span>.<span class="function">fit</span>(x_poly_teste, y_teste)</span>\n\n<span class="comment">#Coeficiente de Determinação Treino</span>\n<span><span class="keyword">coeficiente_determinacao_treino = </span>regressao_polinomial_treino.<span class="function">score</span>(x_poly_treino, y_treino)</span>\n<span><span class="function">print</span>(f"Coeficiente de determinação treino: {{coeficiente_determinacao_treino}}")</span>\n\n<span class="comment">#Coeficiente de Determinação Teste</span>\n<span><span class="keyword">coeficiente_determinacao_teste = </span>regressao_polinomial_teste.<span class="function">score</span>(x_poly_teste, y_teste)</span>\n<span><span class="function">print</span>(f"Coeficiente de determinação teste: {{coeficiente_determinacao_teste}}")</span>\n\n<span class="comment">#Realiza previsões do treino</span>\n<span><span class="keyword">previsoes</span> = regressao_polinomial_treino.<span class="function">predict</span>(x_poly_teste)</span>\n\n<span class="comment">#Biblioteca de métricas </span>\n<span><span class="bib">from </span>sklearn.metrics <span class="bib">import </span>mean_absolute_error, mean_squared_error</span>\n\n<span class="comment">#Erro médio Absoluto</span>\n<span><span class="keyword">erro_medio_absoluto</span> = <span class="function">mean_absolute_error</span>(y_teste, previsoes)</span>\n<span><span class="function">print</span>(f"Erro médio Absoluto: {{erro_medio_absoluto}}")</span>\n\n\n<span class="comment">#Erro quadrático médio</span>\n<span><span class="keyword">erro_quadratico_medio</span>= <span class="function">mean_squared_error</span>(y_teste, previsoes)\n</span><span><span class="function">print</span>(f"Erro quadrático médio: {{erro_quadratico_medio}}")</span>\n\n<span class="comment">#Raiz do erro quadrático médio</span>\n<span><span class="keyword">raiz_erro_quadratico_medio</span>= <span class="function">np.sqrt</span>(mean_squared_error(y_teste, previsoes))</span>\n</span><span><span class="function">print</span>(f"Raiz do erro quadrático médio: {{raiz_erro_quadratico_medio}}")</span>\n"""

    elif(classifier_type == "SUPPORT VECTORS(SVR)"):
        parameters = parameters_forms()
        reg =  SVR(kernel= parameters[0])
        X,y = all_columns()
        code+= f"""\n<span class="keyword">independentes</span><span> = dataframe.iloc[:, :-1]</span>\n<span class="keyword">dependentes</span><span> = dataframe.iloc[:, -1]</span>\n<span class="bib">from</span> sklearn.model_selection <span class="bib">import</span> train_test_split\n<span class="keyword">x_treino, x_teste, y_treino, y_teste =</span> <span class="function">train_test_split(independentes,dependentes, test_size = 0.3, random_state = 0)</span><span class="comment">#Base treino e teste</span>\n\n\n<span class="comment">#Padronizando os valores obrigatoriamente para o funcionamento do algoritmo SVR</span>\n<span><span class="bib">from</span> sklearn.preprocessing <span class="bib">import</span> StandardScaler</span>\n<span><span class="keyword">x_scaler</span> = <span class="function">StandardScaler</span>()</span>\n<span><span class="keyword">y_scaler</span> = <span class="function">StandardScaler</span>()</span>\n\n<span><span class="keyword">x_treino_scaler</span> =  x_scaler.<span class="function">fit_transform</span>(x_treino)</span>\n<span><span class="keyword">y_treino_scaler</span> =  y_scaler.<span class="function">fit_transform</span>(y_treino.values.reshape(-1, 1))</span>\n<span><span class="keyword">x_teste_scaler</span> =  x_scaler.<span class="function">transform</span>(x_teste)</span>\n<span><span class="keyword">y_teste_scaler</span> =  y_scaler.<span class="function">transform</span>(y_teste.values.reshape(-1, 1))</span>\n\n<span class="comment">#Importando biblioteca do SVM(SVR)</span>\n<span><span class="bib">from </span>sklearn.svm <span class="bib">import </span>SVR</span>\n<span><span class="keyword">maquina_vetores_suporte </span><span>= </span><span class="function">SVR</span>(kernel='{parameters[0]}')</span>\n<span><span class="keyword">maquina_vetores_suporte</span>.<span class="function">fit</span>(x_treino_scaler, y_treino_scaler.ravel()) <span class="comment">#.ravel() é para retornar matriz 1D</span></span>\n\n<span class="comment">#Coeficiente de Determinação Treino</span>\n<span><span class="keyword">coeficiente_determinacao_treino = </span>maquina_vetores_suporte.<span class="function">score</span>(x_treino_scaler, y_treino_scaler)</span>\n<span><span class="function">print</span>(f"Coeficiente de determinação treino: {{coeficiente_determinacao_treino}}")</span>\n\n<span class="comment">#Coeficiente de Determinação Teste</span>\n<span><span class="keyword">coeficiente_determinacao_teste = </span>maquina_vetores_suporte.<span class="function">score</span>(x_teste_scaler, y_teste_scaler)</span>\n<span><span class="function">print</span>(f"Coeficiente de determinação teste: {{coeficiente_determinacao_teste}}")</span>\n\n<span class="comment">#Realiza previsões do teste</span>\n<span><span class="keyword">previsoes</span> = maquina_vetores_suporte.<span class="function">predict</span>(x_teste_scaler)</span>\n\n<span class="comment">#Revertendo a transformação para obter os dados reais não escalonados</span>\n<span><span class="keyword">y_teste_inverse</span> = y_scaler.<span class="function">inverse_transform</span>(y_teste_scaler)</span>\n<span><span class="keyword">previsoes_inverse</span> = y_scaler.<span class="function">inverse_transform</span>(previsoes_teste.reshape(-1, 1))</span>\n\n<span class="comment">#Biblioteca de métricas </span>\n<span><span class="bib">from </span>sklearn.metrics <span class="bib">import </span>mean_absolute_error, mean_squared_error</span>\n\n<span class="comment">#Erro médio Absoluto</span>\n<span><span class="keyword">erro_medio_absoluto</span> = <span class="function">mean_absolute_error</span>(y_teste_inverse, previsoes_inverse)</span>\n<span><span class="function">print</span>(f"Erro médio Absoluto: {{erro_medio_absoluto}}")</span>\n\n\n<span class="comment">#Erro quadrático médio</span>\n<span><span class="keyword">erro_quadratico_medio</span>= <span class="function">mean_squared_error</span>(y_teste_inverse, previsoes_inverse)\n</span><span><span class="function">print</span>(f"Erro quadrático médio: {{erro_quadratico_medio}}")</span>\n\n<span class="comment">#Raiz do erro quadrático médio</span>\n<span><span class="keyword">raiz_erro_quadratico_medio</span>= <span class="function">np.sqrt</span>(mean_squared_error(y_teste_inverse, previsoes_inverse))</span>\n</span><span><span class="function">print</span>(f"Raiz do erro quadrático médio: {{raiz_erro_quadratico_medio}}")</span>\n"""

    
    elif(classifier_type == "DECISION TREE"):
        parameters = parameters_forms()
        reg =  DecisionTreeRegressor(max_depth= int(parameters[0]), random_state=int(parameters[1]))
        X,y = all_columns()
        code+= f"""\n<span class="keyword">independentes</span><span> = dataframe.iloc[:, :-1]</span>\n<span class="keyword">dependentes</span><span> = dataframe.iloc[:, -1]</span>\n<span class="bib">from</span> sklearn.model_selection <span class="bib">import</span> train_test_split\n<span class="keyword">x_treino, x_teste, y_treino, y_teste =</span> <span class="function">train_test_split(independentes,dependentes, test_size = 0.3, random_state = 0)</span><span class="comment">#Base treino e teste</span>\n\n<span class="comment">#Importando biblioteca da arvore de decisão </span>\n<span><span class="bib">from </span>sklearn.tree <span class="bib">import </span>DecisionTreeRegressor</span>\n<span><span class="keyword">arvore </span><span>= </span><span class="function">DecisionTreeRegressor</span>(max_depth={parameters[0]},random_state={parameters[1]})</span>\n<span><span class="keyword">arvore</span>.<span class="function">fit</span>(x_treino, y_treino)</span>\n\n<span class="comment">#Coeficiente de Determinação Treino</span>\n<span><span class="keyword">coeficiente_determinacao_treino = </span>arvore.<span class="function">score</span>(x_treino, y_treino)</span>\n<span><span class="function">print</span>(f"Coeficiente de determinação treino: {{coeficiente_determinacao_treino}}")</span>\n\n<span class="comment">#Coeficiente de Determinação Teste</span>\n<span><span class="keyword">coeficiente_determinacao_teste = </span>arvore.<span class="function">score</span>(x_teste, y_teste)</span>\n<span><span class="function">print</span>(f"Coeficiente de determinação teste: {{coeficiente_determinacao_teste}}")</span>\n\n<span class="comment">#Realiza previsões do teste</span>\n<span><span class="keyword">previsoes</span> = arvore.<span class="function">predict</span>(x_teste)</span>\n\n<span class="comment">#Biblioteca de métricas </span>\n<span><span class="bib">from </span>sklearn.metrics <span class="bib">import </span>mean_absolute_error, mean_squared_error</span>\n\n<span class="comment">#Erro médio Absoluto</span>\n<span><span class="keyword">erro_medio_absoluto</span> = <span class="function">mean_absolute_error</span>(y_teste, previsoes)</span>\n<span><span class="function">print</span>(f"Erro médio Absoluto: {{erro_medio_absoluto}}")</span>\n\n\n<span class="comment">#Erro quadrático médio</span>\n<span><span class="keyword">erro_quadratico_medio</span>= <span class="function">mean_squared_error</span>(y_teste, previsoes)\n</span><span><span class="function">print</span>(f"Erro quadrático médio: {{erro_quadratico_medio}}")</span>\n\n<span class="comment">#Raiz do erro quadrático médio</span>\n<span><span class="keyword">raiz_erro_quadratico_medio</span>= <span class="function">np.sqrt</span>(mean_squared_error(y_teste, previsoes))</span>\n</span><span><span class="function">print</span>(f"Raiz do erro quadrático médio: {{raiz_erro_quadratico_medio}}")</span>\n"""

    
    elif(classifier_type == "RANDOM FOREST"):
        parameters = parameters_forms()
        reg = RandomForestRegressor(n_estimators=int(parameters[0]), criterion=parameters[1], max_depth=int(parameters[2]), random_state = int(parameters[3])) 
        X,y = all_columns()
        code+= f"""\n<span class="keyword">independentes</span><span> = dataframe.iloc[:, :-1]</span>\n<span class="keyword">dependentes</span><span> = dataframe.iloc[:, -1]</span>\n<span class="bib">from</span> sklearn.model_selection <span class="bib">import</span> train_test_split\n<span class="keyword">x_treino, x_teste, y_treino, y_teste =</span> <span class="function">train_test_split(independentes,dependentes, test_size = 0.3, random_state = 0)</span><span class="comment">#Base treino e teste</span>\n\n<span class="comment">#Importando biblioteca floresta de arvores aleatorias </span>\n<span><span class="bib">from </span>sklearn.ensemble <span class="bib">import </span>RandomForestRegressor</span>\n<span><span class="keyword">random </span><span>= </span><span class="function">RandomForestRegressor</span>(n_estimators={parameters[0]}, criterion='{parameters[1]}', max_depth={parameters[2]}, random_state = {parameters[3]})</span>\n<span><span class="keyword">random</span>.<span class="function">fit</span>(x_treino, y_treino)</span>\n\n<span class="comment">#Coeficiente de Determinação Treino</span>\n<span><span class="keyword">coeficiente_determinacao_treino = </span>random.<span class="function">score</span>(x_treino, y_treino)</span>\n<span><span class="function">print</span>(f"Coeficiente de determinação treino: {{coeficiente_determinacao_treino}}")</span>\n\n<span class="comment">#Coeficiente de Determinação Teste</span>\n<span><span class="keyword">coeficiente_determinacao_teste = </span>random.<span class="function">score</span>(x_teste, y_teste)</span>\n<span><span class="function">print</span>(f"Coeficiente de determinação teste: {{coeficiente_determinacao_teste}}")</span>\n\n<span class="comment">#Realiza previsões do teste</span>\n<span><span class="keyword">previsoes</span> = random.<span class="function">predict</span>(x_teste)</span>\n\n<span class="comment">#Biblioteca de métricas </span>\n<span><span class="bib">from </span>sklearn.metrics <span class="bib">import </span>mean_absolute_error, mean_squared_error</span>\n\n<span class="comment">#Erro médio Absoluto</span>\n<span><span class="keyword">erro_medio_absoluto</span> = <span class="function">mean_absolute_error</span>(y_teste, previsoes)</span>\n<span><span class="function">print</span>(f"Erro médio Absoluto: {{erro_medio_absoluto}}")</span>\n\n\n<span class="comment">#Erro quadrático médio</span>\n<span><span class="keyword">erro_quadratico_medio</span>= <span class="function">mean_squared_error</span>(y_teste, previsoes)\n</span><span><span class="function">print</span>(f"Erro quadrático médio: {{erro_quadratico_medio}}")</span>\n\n<span class="comment">#Raiz do erro quadrático médio</span>\n<span><span class="keyword">raiz_erro_quadratico_medio</span>= <span class="function">np.sqrt</span>(mean_squared_error(y_teste, previsoes))</span>\n</span><span><span class="function">print</span>(f"Raiz do erro quadrático médio: {{raiz_erro_quadratico_medio}}")</span>\n"""


    elif(classifier_type == "XGBOOST"):
        parameters = parameters_forms()
        reg = XGBRegressor(n_estimators=int(parameters[0]), max_depth=int(parameters[1]), learning_rate=float(parameters[2]), objective=parameters[3], random_state=int(parameters[4]))
        X,y = all_columns()
        code+= f"""\n<span class="keyword">independentes</span><span> = dataframe.iloc[:, :-1]</span>\n<span class="keyword">dependentes</span><span> = dataframe.iloc[:, -1]</span>\n<span class="bib">from</span> sklearn.model_selection <span class="bib">import</span> train_test_split\n<span class="keyword">x_treino, x_teste, y_treino, y_teste =</span> <span class="function">train_test_split(independentes,dependentes, test_size = 0.3, random_state = 0)</span><span class="comment">#Base treino e teste</span>\n\n<span class="comment">#Importando biblioteca do XGBOOST </span>\n<span><span class="bib">from </span>xgboost  <span class="bib">import </span>XGBRegressor</span>\n<span><span class="keyword">regressao_xgboost </span><span>= </span><span class="function">XGBRegressor</span>(n_estimators={parameters[0]}, max_depth={parameters[1]}, learning_rate={parameters[2]}, objective="{parameters[3]}", random_state={parameters[4]})</span>\n<span><span class="keyword">regressao_xgboost</span>.<span class="function">fit</span>(x_treino, y_treino)</span>\n\n<span class="comment">#Coeficiente de Determinação Treino</span>\n<span><span class="keyword">coeficiente_determinacao_treino = </span>regressao_xgboost.<span class="function">score</span>(x_treino, y_treino)</span>\n<span><span class="function">print</span>(f"Coeficiente de determinação treino: {{coeficiente_determinacao_treino}}")</span>\n\n<span class="comment">#Coeficiente de Determinação Teste</span>\n<span><span class="keyword">coeficiente_determinacao_teste = </span>regressao_xgboost.<span class="function">score</span>(x_teste, y_teste)</span>\n<span><span class="function">print</span>(f"Coeficiente de determinação teste: {{coeficiente_determinacao_teste}}")</span>\n\n<span class="comment">#Realiza previsões do teste</span>\n<span><span class="keyword">previsoes</span> = regressao_xgboost.<span class="function">predict</span>(x_teste)</span>\n\n<span class="comment">#Biblioteca de métricas </span>\n<span><span class="bib">from </span>sklearn.metrics <span class="bib">import </span>mean_absolute_error, mean_squared_error</span>\n\n<span class="comment">#Erro médio Absoluto</span>\n<span><span class="keyword">erro_medio_absoluto</span> = <span class="function">mean_absolute_error</span>(y_teste, previsoes)</span>\n<span><span class="function">print</span>(f"Erro médio Absoluto: {{erro_medio_absoluto}}")</span>\n\n\n<span class="comment">#Erro quadrático médio</span>\n<span><span class="keyword">erro_quadratico_medio</span>= <span class="function">mean_squared_error</span>(y_teste, previsoes)\n</span><span><span class="function">print</span>(f"Erro quadrático médio: {{erro_quadratico_medio}}")</span>\n\n<span class="comment">#Raiz do erro quadrático médio</span>\n<span><span class="keyword">raiz_erro_quadratico_medio</span>= <span class="function">np.sqrt</span>(mean_squared_error(y_teste, previsoes))</span>\n</span><span><span class="function">print</span>(f"Raiz do erro quadrático médio: {{raiz_erro_quadratico_medio}}")</span>\n"""


    elif(classifier_type == "LIGHT GBM"):
        parameters = parameters_forms()
        reg = lgbm.LGBMRegressor(num_leaves=int(parameters[0]), max_depth=int(parameters[1]), learning_rate=float(parameters[2]), n_estimators=int(parameters[3]), random_state=int(parameters[4]))
        X,y = all_columns()
        code+= f"""\n<span class="keyword">independentes</span><span> = dataframe.iloc[:, :-1]</span>\n<span class="keyword">dependentes</span><span> = dataframe.iloc[:, -1]</span>\n<span class="bib">from</span> sklearn.model_selection <span class="bib">import</span> train_test_split\n<span class="keyword">x_treino, x_teste, y_treino, y_teste =</span> <span class="function">train_test_split(independentes,dependentes, test_size = 0.3, random_state = 0)</span><span class="comment">#Base treino e teste</span>\n\n<span class="comment">#Importando biblioteca do LIGHTGBM, execute esse comando uma vez no collab(!pip install lightgbm) ou terminal (pip install lightgbm) </span>\n<span><span class="bib">import </span>lightgbm  <span class="bib">as </span>lgb</span>\n<span><span class="keyword">regressao_lightgbm </span><span>= </span><span class="function">lgb.LGBMRegressor</span>(num_leaves={parameters[0]}, max_depth={parameters[1]}, learning_rate={parameters[2]}, n_estimators={parameters[3]}, random_state={parameters[4]})</span>\n<span><span class="keyword">regressao_lightgbm</span>.<span class="function">fit</span>(x_treino, y_treino)</span>\n\n<span class="comment">#Coeficiente de Determinação Treino</span>\n<span><span class="keyword">coeficiente_determinacao_treino = </span>regressao_lightgbm.<span class="function">score</span>(x_treino, y_treino)</span>\n<span><span class="function">print</span>(f"Coeficiente de determinação treino: {{coeficiente_determinacao_treino}}")</span>\n\n<span class="comment">#Coeficiente de Determinação Teste</span>\n<span><span class="keyword">coeficiente_determinacao_teste = </span>regressao_lightgbm.<span class="function">score</span>(x_teste, y_teste)</span>\n<span><span class="function">print</span>(f"Coeficiente de determinação teste: {{coeficiente_determinacao_teste}}")</span>\n\n<span class="comment">#Realiza previsões do teste</span>\n<span><span class="keyword">previsoes</span> = regressao_lightgbm.<span class="function">predict</span>(x_teste)</span>\n\n<span class="comment">#Biblioteca de métricas </span>\n<span><span class="bib">from </span>sklearn.metrics <span class="bib">import </span>mean_absolute_error, mean_squared_error</span>\n\n<span class="comment">#Erro médio Absoluto</span>\n<span><span class="keyword">erro_medio_absoluto</span> = <span class="function">mean_absolute_error</span>(y_teste, previsoes)</span>\n<span><span class="function">print</span>(f"Erro médio Absoluto: {{erro_medio_absoluto}}")</span>\n\n\n<span class="comment">#Erro quadrático médio</span>\n<span><span class="keyword">erro_quadratico_medio</span>= <span class="function">mean_squared_error</span>(y_teste, previsoes)\n</span><span><span class="function">print</span>(f"Erro quadrático médio: {{erro_quadratico_medio}}")</span>\n\n<span class="comment">#Raiz do erro quadrático médio</span>\n<span><span class="keyword">raiz_erro_quadratico_medio</span>= <span class="function">np.sqrt</span>(mean_squared_error(y_teste, previsoes))</span>\n</span><span><span class="function">print</span>(f"Raiz do erro quadrático médio: {{raiz_erro_quadratico_medio}}")</span>\n"""

    
    elif(classifier_type == "CATBOOST"):
        parameters = parameters_forms()
        reg = CatBoostRegressor(iterations=int(parameters[0]), learning_rate=float(parameters[1]), depth = int(parameters[2]), random_state = int(parameters[3]))
        X,y = all_columns()
        code+= f"""\n<span class="keyword">independentes</span><span> = dataframe.iloc[:, :-1]</span>\n<span class="keyword">dependentes</span><span> = dataframe.iloc[:, -1]</span>\n<span class="bib">from</span> sklearn.model_selection <span class="bib">import</span> train_test_split\n<span class="keyword">x_treino, x_teste, y_treino, y_teste =</span> <span class="function">train_test_split(independentes,dependentes, test_size = 0.3, random_state = 0)</span><span class="comment">#Base treino e teste</span>\n\n<span class="comment">#Importando biblioteca do CATBOOST, execute esse comando uma vez no collab(!pip install catboost) ou terminal (pip install catboost) </span>\n<span><span class="bib">from </span>catboost.core  <span class="bib">import </span>CatBoostRegressor</span>\n<span><span class="keyword">regressao_catboost </span><span>= </span><span class="function">CatBoostRegressor</span>(iterations={parameters[0]}, learning_rate={parameters[1]}, depth = {parameters[2]}, random_state = {parameters[3]})</span>\n<span><span class="keyword">regressao_catboost</span>.<span class="function">fit</span>(x_treino, y_treino)</span>\n\n<span class="comment">#Coeficiente de Determinação Treino</span>\n<span><span class="keyword">coeficiente_determinacao_treino = </span>regressao_catboost.<span class="function">score</span>(x_treino, y_treino)</span>\n<span><span class="function">print</span>(f"Coeficiente de determinação treino: {{coeficiente_determinacao_treino}}")</span>\n\n<span class="comment">#Coeficiente de Determinação Teste</span>\n<span><span class="keyword">coeficiente_determinacao_teste = </span>regressao_catboost.<span class="function">score</span>(x_teste, y_teste)</span>\n<span><span class="function">print</span>(f"Coeficiente de determinação teste: {{coeficiente_determinacao_teste}}")</span>\n\n<span class="comment">#Realiza previsões do teste</span>\n<span><span class="keyword">previsoes</span> = regressao_catboost.<span class="function">predict</span>(x_teste)</span>\n\n<span class="comment">#Biblioteca de métricas </span>\n<span><span class="bib">from </span>sklearn.metrics <span class="bib">import </span>mean_absolute_error, mean_squared_error</span>\n\n<span class="comment">#Erro médio Absoluto</span>\n<span><span class="keyword">erro_medio_absoluto</span> = <span class="function">mean_absolute_error</span>(y_teste, previsoes)</span>\n<span><span class="function">print</span>(f"Erro médio Absoluto: {{erro_medio_absoluto}}")</span>\n\n\n<span class="comment">#Erro quadrático médio</span>\n<span><span class="keyword">erro_quadratico_medio</span>= <span class="function">mean_squared_error</span>(y_teste, previsoes)\n</span><span><span class="function">print</span>(f"Erro quadrático médio: {{erro_quadratico_medio}}")</span>\n\n<span class="comment">#Raiz do erro quadrático médio</span>\n<span><span class="keyword">raiz_erro_quadratico_medio</span>= <span class="function">np.sqrt</span>(mean_squared_error(y_teste, previsoes))</span>\n</span><span><span class="function">print</span>(f"Raiz do erro quadrático médio: {{raiz_erro_quadratico_medio}}")</span>\n"""


    
    #Base treino e teste 
    x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.3, random_state = 0)
    
    
    
    if classifier_type in name_Regression_fit:
        determinationCoefficientTraining,determinationCoefficientTest,absolute,MeanSquaredError = train_fit(reg,x_teste, x_treino, y_teste, y_treino)
        

    elif classifier_type in name_Regression_poly:
        determinationCoefficientTraining,determinationCoefficientTest,absolute,MeanSquaredError = train_poly(reg,x_teste, x_treino, y_teste, y_treino)
    
    elif classifier_type in name_Regression_SVR:
        determinationCoefficientTraining,determinationCoefficientTest,absolute,MeanSquaredError = train_SVR(reg,x_teste, x_treino, y_teste, y_treino)


    response = {
        "determinationCoefficientTraining": round((determinationCoefficientTraining *100), 3), 
        "determinationCoefficientTest": round((determinationCoefficientTest*100),3), 
        "abs":round((absolute), 3), 
        "MeanSquaredError":round((MeanSquaredError), 3),
        "code":code
    }


    # print(f"features: {features}")
    if deployBoolean == "true":
        val = deployBooleanTrue(csv_data, csv_tranform, csv_deploy, meu_dicionario, meu_dicionarioencoder, reg)
        val = [round(valor, 2) for valor in val]
        response["prediction"] = val
    

    if crossVal == "true":
        x = csv_data.iloc[:, :-1]
        y = csv_data.iloc[:, -1]

        print("------------------____________-------------------")
        print(X)
        print(y)
        crossValValue = crossValidation(reg,x,y)
        response["crossVal"] = round(crossValValue, 3)     

    return jsonify(response)



@app.route('/graphicAnalysisPost', methods=['POST'])
def submit_selections_graphicAnalysis():
    typeGraphic = request.form.get('typeGraphic')
    print(f"typeGraphic {typeGraphic}")
    csv_file = request.files.get('csvFile')
    selections = request.form.get('selections')
    separator = request.form.get('separator')
    selections = json.loads(selections)
    print(selections)
    # Processar o CSV
    csv_data = pd.read_csv(io.BytesIO(csv_file.read()), sep=separator, encoding='utf-8')
    print(csv_data)
    code = ""

    
    
    def createGraph(key, values, data):
        if values == "histogramas":
            graf = go.Figure(data=[go.Histogram(x=csv_data[key], nbinsx=60)])
            graf.update_layout(width=800, height=500, title_text='Distribuição por '+ key, paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = graf.to_json()
            data[key] = graf_json
            code = f"""<span class="keyword">histograma </span><span>= px.</span><span class="function">histogram</span><span class="object">(dataframe, x= "{key}", title= 'Histograma de {key}', width= 800, height= 500)</span>\n<span class="keyword">histograma.</span><span class="function">update_layout</span><span>(<span class="object">paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)'</span>)</span>\n<span><span class="keyword">histograma</span>.<span class="function">show</span>()</span>"""


        elif values == "gráficos de pizza":
            counts = csv_data[key].value_counts()
            print(f"counts: {counts}\ncounts.index: {counts.index}\ncounts.values: {counts.values}")
            graf = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values)])
            graf.update_layout(width=800, height=500, title_text='Distribuição por '+ key, paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = graf.to_json()
            data[key] = graf_json
            code = f"""<span class="keyword">grafico_pizza </span><span>= px.</span><span class="function">pie</span><span class="object">(dataframe, names= "{key}", title= 'Gráfico de Pizza de {key}', width= 800, height= 500)</span>\n<span class="keyword">grafico_pizza.</span><span class="function">update_layout</span><span>(<span class="object">paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)'</span>)</span>\n<span><span class="keyword">grafico_pizza</span>.<span class="function">show</span>()</span>"""



        elif values == "gráficos de linha":
            counts = csv_data[key].value_counts()
            graf = go.Figure(data=[go.Scatter(x=counts.index, y=counts.values, mode='lines+markers', name='Distribuição por ' + key)])
            graf.update_layout(width=800, height=500, title_text='Distribuição por ' + key,paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = graf.to_json()
            data[key] = graf_json
            code = f"""<span><span class="keyword">dataframe_agrupado</span> = dataframe.<span class="function">groupby</span>(["{key}"]).size().reset_index(name='soma')</span>\n<span class="keyword">grafico_linha </span><span>= px.</span><span class="function">line</span><span class="object">(dataframe_agrupado, x= "{key}", y="soma", title= 'Gráfico de linha de {key}', width= 800, height= 500)</span>\n<span class="keyword">grafico_linha.</span><span class="function">update_layout</span><span>(<span class="object">paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)'</span>)</span>\n<span><span class="keyword">grafico_linha</span>.<span class="function">show</span>()</span>"""

        elif values == "gráficos de barras":
            counts = csv_data[key].value_counts()
            graf = go.Figure(data=[go.Bar(x=counts.index,y=counts.values,name='Distribuição por ' + key)])
            graf.update_layout(title='Distribuição por ' + key, xaxis_title='Categoria', yaxis_title='Frequência', width=800, height=500,paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = graf.to_json()
            data[key] = graf_json
            code = f"""<span><span class="keyword">dataframe_agrupado</span> = dataframe.<span class="function">groupby</span>(["{key}"]).size().reset_index(name='soma')</span>\n<span class="keyword">grafico_barras</span><span>= px.</span><span class="function">bar</span><span class="object">(dataframe_agrupado, x= "{key}", y= "soma", title= 'Gráfico de barras de {key}', width= 800, height= 500)</span>\n<span class="keyword">grafico_barras.</span><span class="function">update_layout</span><span>(<span class="object">paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)'</span>)</span>\n<span><span class="keyword">grafico_barras</span>.<span class="function">show</span>()</span>"""


        elif values == "gráficos de dispersão":
            counts = csv_data[key].value_counts()
            graf = go.Figure(data=[go.Scatter(x=counts.index, y=counts.values, mode='markers', name='Distribuição por ' + key)])
            graf.update_layout(width=800, height=500, title='Distribuição por ' + key, xaxis_title='X',yaxis_title='Y', paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = graf.to_json()
            data[key] = graf_json
            code = f"""<span><span class="keyword">dataframe_agrupado</span> = dataframe.<span class="function">groupby</span>(["{key}"]).size().reset_index(name='soma')</span>\n<span class="keyword">grafico_dispersao</span><span>= px.</span><span class="function">scatter</span><span class="object">(dataframe_agrupado, x= "{key}", y= "soma", title= 'Gráfico de dispersão de {key}', width= 800, height= 500)</span>\n<span class="keyword">grafico_dispersao.</span><span class="function">update_layout</span><span>(<span class="object">paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)'</span>)</span>\n<span><span class="keyword">grafico_dispersao</span>.<span class="function">show</span>()</span>"""


        elif values == "boxplot":
            counts = csv_data[key].value_counts()
            graf = go.Figure(data=[go.Box( y= counts.values ,name='Distribuição por ' + key)])
            graf.update_layout(width=800, height=500,  yaxis_title='Valores', title='Distribuição por ' + key, paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
            graf_json = graf.to_json()
            data[key] = graf_json
            code = f"""<span class="keyword">grafico_boxplot</span><span>= px.</span><span class="function">box</span><span class="object">(dataframe, x= "{key}", title= 'Gráfico BoxPlot de {key}', width= 800, height= 500)</span>\n<span class="keyword">grafico_boxplot.</span><span class="function">update_layout</span><span>(<span class="object">paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)'</span>)</span>\n<span><span class="keyword">grafico_boxplot</span>.<span class="function">show</span>()</span>"""


        return data,code
    

    def createGraph2Var(graphicType, var1, var2, data):
        code = f"""<span class= "comment">#Leitura do arquivo csv</span>\n<span class="bib">import</span> pandas <span class="bib">as</span> pd\n<span>dataframe = pd.read_csv(</span><span class= "link">'Adicione o caminho para o seu csv',sep ='{separator}', encoding = 'utf-8'</span>)</span>\n\n<span class= "comment">#Biblioteca grafica</span>\n<span class="bib">import</span> plotly.express <span class="bib">as</span> px\n"""
        # Agrupando por idade e sexo e contando a quantidade de cada combinação
        df_grouped = csv_data.groupby([var1, var2]).size().reset_index(name='soma')
        print(f"agrupados: {df_grouped}")

        # Encontrando as categorias únicas
        categorias = df_grouped[var1].unique()

        # Gerando um mapa de cores automaticamente
        cores = px.colors.qualitative.Plotly  # Lista de cores padrão
        color_discrete_map = {cat: cores[i % len(cores)] for i, cat in enumerate(categorias)}
        
        if graphicType == "Histogramas":
            # Criando o gráfico de histograma
            hist_fig = px.histogram(csv_data, x=var1, color=var2, title=f'Histograma de {var1} por {var2}', color_discrete_map=color_discrete_map, width=800, height=500)
            hist_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            # Convertendo o gráfico para JSON
            graf_json = hist_fig.to_json()         
            data[graphicType] = graf_json
            code+= f"""<span class="keyword">histograma </span><span>= px.</span><span class="function">histogram</span><span class="object">(dataframe, x= "{var1}", color= "{var2}", title= 'Histograma de {var1} por {var2}', width= 800, height= 500)</span>\n<span class="keyword">histograma.</span><span class="function">update_layout</span><span>(<span class="object">paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)'</span>)</span>\n<span><span class="keyword">histograma</span>.<span class="function">show</span>()</span>"""

        
        elif(graphicType == "Gráficos de Barras"):
            hist_fig = px.bar(df_grouped, x= var1, y='soma', color=var2,title=f'Gráficos de barras de {var1} por {var2}', barmode='stack',color_discrete_map=color_discrete_map, width=800, height=500)
            hist_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = hist_fig.to_json()
            data[graphicType] = graf_json
            code+= f"""\n<span class="comment">#Agrupando por {var1} e {var2} e contando a quantidade de cada combinação</span>\n<span class="keyword">df_grouped</span><span>= dataframe.</span><span class="function">groupby</span><span class="object">(["{var1}", "{var2}"]).size().reset_index(name='soma')</span>\n<span class="function">print</span><span>(f"agrupados: {{df_grouped}}")</span>\n\n<span class="comment">#Gerando Gráfico</span>\n<span class="keyword">grafico_barra</span><span>= px.</span><span class="function">bar</span><span class="object">(df_grouped, x= '{var1}', y='soma', color='{var2}',title='Gráficos de barras de {var1} por {var2}',barmode='stack',width=800, height=500)</span>\n<span class="keyword">grafico_barra.</span><span class="function">update_layout</span><span>(<span class="object">paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)'</span>)</span>\n<span><span class="keyword">grafico_barra</span>.<span class="function">show</span>()</span>"""

        
        elif(graphicType == "Gráficos de Linha"):
            hist_fig = px.line(df_grouped, x= var1, y='soma', color=var2,title=f'Gráficos de linha de {var1} por {var2}',color_discrete_map=color_discrete_map, width=800, height=500)
            hist_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = hist_fig.to_json()
            data[graphicType] = graf_json
            code+= f"""\n<span class="comment">#Agrupando por {var1} e {var2} e contando a quantidade de cada combinação</span>\n<span class="keyword">df_grouped</span><span>= dataframe.</span><span class="function">groupby</span><span class="object">(["{var1}", "{var2}"]).size().reset_index(name='soma')</span>\n<span class="function">print</span><span>(f"agrupados: {{df_grouped}}")</span>\n\n<span class="comment">#Gerando Gráfico</span>\n<span class="keyword">grafico_linha</span><span>= px.</span><span class="function">line</span><span class="object">(df_grouped, x= '{var1}', y='soma', color='{var2}',title='Gráficos de linhas de {var1} por {var2}',width=800, height=500)</span>\n<span class="keyword">grafico_linha.</span><span class="function">update_layout</span><span>(<span class="object">paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)'</span>)</span>\n<span><span class="keyword">grafico_linha</span>.<span class="function">show</span>()</span>"""


        elif(graphicType == "Gráficos de Pizza"):
            hist_fig = px.pie(df_grouped,names=var1,values='soma', color=var2,title=f'Gráficos de pizza de {var1} por {var2}', color_discrete_map=color_discrete_map, width=800, height=500)
            hist_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = hist_fig.to_json()
            data[graphicType] = graf_json
            code+= f"""\n<span class="comment">#Agrupando por {var1} e {var2} e contando a quantidade de cada combinação</span>\n<span class="keyword">df_grouped</span><span>= dataframe.</span><span class="function">groupby</span><span class="object">(["{var1}", "{var2}"]).size().reset_index(name='soma')</span>\n<span class="function">print</span><span>(f"agrupados: {{df_grouped}}")</span>\n\n<span class="comment">#Gerando Gráfico</span>\n<span class="keyword">grafico_pizza</span><span>= px.</span><span class="function">pie</span><span class="object">(df_grouped, x= '{var1}', y='soma', color='{var2}',title='Gráficos de pizza de {var1} por {var2}',width=800, height=500)</span>\n<span class="keyword">grafico_pizza.</span><span class="function">update_layout</span><span>(<span class="object">paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)'</span>)</span>\n<span><span class="keyword">grafico_pizza</span>.<span class="function">show</span>()</span>"""

        
        elif(graphicType == "Gráficos de Dispersão"):
            hist_fig = px.scatter(df_grouped,x= var1, y='soma',color=var2,title=f'Gráficos de dispersão de {var1} por {var2}', color_discrete_map=color_discrete_map, width=800, height=500)
            hist_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = hist_fig.to_json()
            data[graphicType] = graf_json
            code+= f"""\n<span class="comment">#Agrupando por {var1} e {var2} e contando a quantidade de cada combinação</span>\n<span class="keyword">df_grouped</span><span>= dataframe.</span><span class="function">groupby</span><span class="object">(["{var1}", "{var2}"]).size().reset_index(name='soma')</span>\n<span class="function">print</span><span>(f"agrupados: {{df_grouped}}")</span>\n\n<span class="comment">#Gerando Gráfico</span>\n<span class="keyword">grafico_dispersao</span><span>= px.</span><span class="function">scatter</span><span class="object">(df_grouped, x= '{var1}', y='soma', color='{var2}',title='Gráficos de dispersão de {var1} por {var2}',width=800, height=500)</span>\n<span class="keyword">grafico_dispersao.</span><span class="function">update_layout</span><span>(<span class="object">paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)'</span>)</span>\n<span><span class="keyword">grafico_dispersao</span>.<span class="function">show</span>()</span>"""

        print(f"O codigo é {code}")

        return data, code
    
    
    
    def typeGraph(selections):
        code = ""
        codecopy = "" 
        data = dict()
        for key, values in selections.items():
          print(f"Chave: {key}\nValor: {values}\n")
          if(typeGraphic == "true"):
            codecopy = code +"\n\n"
            data,code = createGraph(key, values, data)
            code = codecopy + code  
            print(f"{code}\n")    
        code = f"""<span class= "comment">#Leitura do arquivo csv</span>\n<span class="bib">import</span> pandas <span class="bib">as</span> pd\n<span>dataframe = pd.read_csv(</span><span class= "link">'Adicione o caminho para o seu csv',sep ='{separator}', encoding = 'utf-8'</span>)</span>\n\n<span class= "comment">#Biblioteca grafica</span>\n<span class="bib">import</span> plotly.express <span class="bib">as</span> px\n"""+code  
        return data, code
    
    def typeGraph2Var(selections):
        data = dict()
        graphicType = list(selections.values())[0]
        var1 = list(selections.values())[1]
        var2 = list(selections.values())[2]
        print(f"v1 {var1} v2 {var2}")
        data,code = createGraph2Var(graphicType, var1, var2, data)
        print(f"Data {data}")
        return data, code
    
    
    if(typeGraphic == "true"):
        data,code = typeGraph(selections)
        print(f" Tipo : {type(data)}")
        return jsonify({"data":data, "code":code})

    elif(typeGraphic == "false"):
        data,code = typeGraph2Var(selections)
        print(f" Tipo : {type(data)}")
        return jsonify({"data":data, "code":code})



@app.route('/associationRulesPost', methods=['POST'])
def submit_selections_associationRules():
    typeGraphic = request.form.get('typeGraphic')
    print(f"typeGraphic {typeGraphic}")
    csv_file = request.files.get('csvFile')
    selections = request.form.get('selections')
    separator = request.form.get('separator')
    selections = json.loads(selections)
    print(selections)
    # Processar o CSV
    csv_data = pd.read_csv(io.BytesIO(csv_file.read()), sep=separator, encoding='utf-8')
    print(csv_data)
    
    def listTranform():
        var1 = list(selections.values())[1]
        var2 = list(selections.values())[2]
        print(var1, var2)
        dados = csv_data.groupby(var1)[var2].apply(list).tolist()
        dados = pd.DataFrame(dados)
        return dados
    
    def applyEclat():
        data = listTranform()
        eclat = ECLAT(data = data, verbose = True)
        df2 = eclat.df_bin
        return df2

    def generatingAssociation(df2):
       associacao = apriori(df2, min_support=0.05, use_colnames=True)
       associacao.sort_values("support", ascending=False).head(10)
       metric = list(selections.values())[0]
       regras = association_rules(associacao, metric=metric)
       print(regras)
       resultado = regras.sort_values("support", ascending=False)
       return resultado  
    
    
    df2 = applyEclat()
    result = generatingAssociation(df2)
    print(result)
    result = result.head(20).to_html(classes='table table-striped', border=0)
    code = f"""<span class= "comment">#Leitura do arquivo csv</span>\n<span class="bib">import</span> pandas <span class="bib">as</span> pd\n<span>dataframe = pd.read_csv(</span><span class= "link">'Adicione o caminho para o seu csv',sep ='{separator}', encoding = 'utf-8'</span>)</span>\n\n<span class="comment">#Agrupa os dados</span>\n<span><span class="keyword">dados</span>= dataframe.<span class="function">groupby</span>("{list(selections.values())[1]}")["{list(selections.values())[2]}"].apply(list).tolist()</span>\n\n<span class="comment">#Converte a lista de listas em um DataFrame do pandas</span>\n<span class="keyword">dados</span><span>= pd.</span><span class="function">DataFrame</span><span>(dados)</span>\n\n<span class="comment">#(!pip install pyECLAT) no colab ou (pip install pyECLAT) no terminal</span>\n<span class="bib">from </span><span>pyECLAT </span><span class="bib">import </span><span> ECLAT</span>\n\n<span class="comment">#Cria uma instância do algoritmo ECLAT, passando os dados e ativando a verbosidade para exibir mensagens de progresso</span>\n<span><span class="keyword">eclat</span>= <span class="function">ECLAT</span>(data = dados, verbose = True)</span>\n\n<span class="comment">#Gera o DataFrame binarizado a partir dos dados processados pelo ECLAT</span>\n<span><span class="keyword">dataframe_eclat</span>= eclat.<span class="function">df_bin</span></span>\n\n<span class="comment">#Importa as funções apriori e association_rules da biblioteca mlxtend.frequent_patterns</span>\n<span><span class="bib">from </span>mlxtend.frequent_patterns <span class="bib">import </span><span class="function">apriori, association_rules</span></span>\n\n<span class="comment">#Aplica o algoritmo apriori no DataFrame binarizado com suporte mínimo de 0.05, mantendo os nomes das colunas</span>\n<span><span class="keyword">associacao</span>= <span class="function">apriori</span>(dataframe_eclat, min_support=0.05, use_colnames=True)</span>\n\n<span class="comment">#Ordena as regras de associação pelo suporte em ordem decrescente e exibe as 10 principais</span>\n<span><span class="link">associacao</span>.<span class="function">sort_values</span>("support", ascending=False).head(10)</span>\n\n<span class="comment">#Gera as regras de associação com base na métrica "{list(selections.values())[0]}"</span>\n<span><span class="keyword">regras</span>= <span class="function">association_rules</span>(associacao, metric="{list(selections.values())[0]}")</span>\n\n<span class="comment">#Imprime as regras geradas</span>\n<span><span class="function">print</span>(regras)</span>\n\n<span class="comment">#Ordena as regras pelo suporte em ordem decrescente e imprime as 20 principais</span>\n<span><span class="keyword">resultado</span>= regras.<span class="function">sort_values</span>("support", ascending=False)</span>\n\n<span><span class="function">print</span>(resultado.head(20))</span>\n\n"""
    return jsonify({"dados": result, "code":code})


   

@app.route("/chat", methods=["POST"])
def chat():
    

    user_message = request.json.get("message")
    print(user_message)
    if not user_message:
        return jsonify({"error": "Nenhuma mensagem recebida."}), 400
    

    chat = model.start_chat(history=[])
    
    # Envia a mensagem e obtém a resposta do bot
    response = chat.send_message(user_message)
    
    # A resposta do modelo é armazenada com o papel "model"
    bot_reply = response.text
   
    
    # Converte a resposta do bot para markdown
    bot_reply = markdown.markdown(bot_reply)
    
    return jsonify({
        "user_message": user_message,
        "bot_reply": bot_reply,
        "timestamp": datetime.datetime.now().isoformat()
    })



if __name__ == "__main__":
    app.run(debug=True)
