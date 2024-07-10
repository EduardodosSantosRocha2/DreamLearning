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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

#Biblioteca Eclat Regras de associaçao(pip install pyECLAT), Apriori (pip install mlxtend)
from pyECLAT import ECLAT
from mlxtend.frequent_patterns import apriori, association_rules

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)



@app.route("/")
def home():
    return render_template("home.html")

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


# Rota para fazer previsões
@app.route("/predict", methods=["POST"])
def predict():
    # Recebe os dados enviados pelo formulário
    classifier_type = request.form["classifier"]
    print(classifier_type)
    csv_file = request.files["csv_file"]      
    separator = request.form.get('separator')
    # Lê o arquivo CSV
    csv_data = pd.read_csv(io.BytesIO(csv_file.read()),sep = separator, encoding = 'utf-8') 
    csv_tranform = csv_data.copy()

    deployBoolean = request.form.get('deployBoolean')
    if deployBoolean == "true":
        csv_file_deploy = request.files["csv_deploy"]
        csv_deploy = pd.read_csv(io.BytesIO(csv_file_deploy.read()), sep= separator, encoding='utf-8')
        print(f"-----------------------------------------------------")
        csv_deploy = csv_deploy.to_numpy().tolist()
        print(csv_deploy)
        print(f"-----------------------------------------------------")


    #Aplica o labelencolder
    meu_dicionario = {}
    meu_dicionarioencoder = {}

    dicionarioParametros = {}
    clf  = ""

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

    # Separa as características e o alvo
    X = csv_data.iloc[:, :-1]
    y = csv_data.iloc[:, -1]

    #Base treino e teste 
    x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.3, random_state = 0)
    
   
    

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

    name_Classifier_fit = ["random_forest", "svm","logistics_regression","knn","decision_tree","xgboost"]
    name_Classifier_predict = ["lightgbm"]
    # Escolhe o classificador com base na escolha do usuário
    if classifier_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=int(parameters[0]), criterion=parameters[1],
                                     random_state=int(parameters[2]), max_depth=int(parameters[3]))
    elif classifier_type == "svm":
        clf = SVC(kernel=parameters[0], random_state=int(parameters[1]), C=int(parameters[2]))
    elif classifier_type == "logistics_regression":
        clf = LogisticRegression(random_state=int(parameters[0]), max_iter=int(parameters[1]), penalty=parameters[2],
                                 tol=float(parameters[3]), C=int(parameters[4]), solver=parameters[5])
    elif classifier_type == "knn":
        clf = KNeighborsClassifier(n_neighbors=int(parameters[0]), metric=parameters[1], p=int(parameters[2]))
    elif classifier_type == "decision_tree":
        clf = DecisionTreeClassifier(criterion=parameters[0], random_state=int(parameters[1]),
                                     max_depth=int(parameters[2]))
    elif classifier_type == "xgboost":
        clf = XGBClassifier(max_depth=int(parameters[0]), learning_rate=float(parameters[1]),
                            n_estimators=int(parameters[2]), objective=parameters[3], random_state=int(parameters[4]))
    
    elif classifier_type == "lightgbm":
        # Configurando o dataset de treino
        dataset = lgbm.Dataset(x_treino, label=y_treino)
        dicionarioParametros = {'num_leaves': int(parameters[0]),  # número de folhas
                                'objective': parameters[1],       # classificação Binária
                                'max_depth': int(parameters[2]),
                                'learning_rate': float(parameters[3]),
                                'max_bin': int(parameters[4])}

        # Treinamento do modelo
        clf = lgbm.train(dicionarioParametros, dataset, num_boost_round=int(parameters[5]))

        # Certificando-se que o treinamento foi completado
        clf.model_from_string(clf.model_to_string())  # Forçando a serialização/deserialização para sincronização

        # Código a ser executado após o treinamento


    
    elif classifier_type == "catboost":
        categoricas = []
        clf = CatBoostClassifier(task_type=parameters[0], iterations=int(parameters[1]), learning_rate=float(parameters[2]), depth = int(parameters[3]), random_state = int(parameters[4]),
                              eval_metric=int(parameters[5]))
    else:
        return jsonify({"error": "Classificador inválido."}), 400

    print("Ate aqui foi")

    def train_fit(clf,x_teste, x_treino, y_teste, y_treino):
        print("chegou na funçao train_fit")
        # Treina o modelo
        clf.fit(x_treino, y_treino)
        #Avaliação do algoritmo
        forecast_test = clf.predict(x_teste)
        #Avaliação de treino
        forecast_training = clf.predict(x_treino)
        return forecast_test, forecast_training

    
    def train_predict(clf,x_teste, x_treino, y_teste, y_treino):
        # Treina o modelo
        forecast_test = clf.predict(x_teste)
        #tranformando as previsoes_lgbm em 0 e 1
        for i in range(len(forecast_test)):
            if forecast_test[i] >=0.5:
                forecast_test[i] = 1
            else:
                forecast_test[i]=0
            
        #Avaliação de treino
        forecast_training = clf.predict(x_treino)
        #tranformando as previsoes_lgbm em 0 e 1
        for i in range(len(forecast_training)):
            if forecast_training[i] >=0.5:
                forecast_training[i] = 1
            else:
                forecast_training[i]=0
        return forecast_test, forecast_training
    
    def print_accuracy_score_test(forecast_test, y_teste):
        accuracy_test = accuracy_score(y_teste, forecast_test)*100
        print(f"Acuracia: {(accuracy_test):.5f}%\n")
        print(confusion_matrix(y_teste, forecast_test))
        return accuracy_test

    

    def print_accuracy_score_traning(forecast_training,y_treino):
        accuracy_training = accuracy_score(y_treino, forecast_training)*100
        print(f"Acuracia treino: {(accuracy_training):.5f}%\n")
        print(confusion_matrix(y_treino, forecast_training))
        return accuracy_training
        
    

    if classifier_type in name_Classifier_fit:
        forecast_test, forecast_training = train_fit(clf,x_teste, x_treino, y_teste, y_treino)
        

    elif classifier_type in name_Classifier_predict:
        forecast_test, forecast_training = train_predict(clf,x_teste, x_treino, y_teste, y_treino)
    
    accuracy_test =  print_accuracy_score_test(forecast_test, y_teste)
    accuracy_training =  print_accuracy_score_traning(forecast_training,y_treino)

    # def is_not_nan(value):
    #     return isinstance(value, (int, float)) and not math.isnan(value)

    
    # def is_not_nan(value):
    #     try:
    #         # Tenta converter o valor para float
    #         float_value = float(value)
    #     except ValueError:
    #         # Se a conversão falhar, não é um número válido
    #         return False
    #     # Verifica se o valor convertido é NaN
    #     return not math.isnan(float_value)

    
    if deployBoolean == "true":
        # # Recebe as características do formulário
        # features = []
        # for i in range(1, X.shape[1] + 1):
        #     form_value = request.form[f"feature{i}"]
        #     if is_not_nan(form_value):
        #         feature = float(form_value)
        #         print(f"feature if {feature}")
        #     else:
        #         feature = form_value
        #         print(f"feature else {feature}")
        #     features.append(feature)

        # print(f"features aaaaa: {features}")
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
        prediction = clf.predict(deploy)
        
        
        if(csv_tranform.iloc[:, -1].dtype == 'object'):
            # Convertendo as chaves do dicionário em uma lista
            lista_chaves = list(meu_dicionarioencoder.keys())

            # Obtendo a última chave
            ultima_chave = lista_chaves[-1]

            # Encontrando o índice da previsão na lista correspondente à última chave
            indice_predicao = meu_dicionarioencoder[ultima_chave].index(prediction[0])

            val = meu_dicionario[ultima_chave][indice_predicao]
        
        else: 
            val = prediction.tolist()
    
        return jsonify({"prediction": f"{val}","accuracy_test": round((accuracy_test), 3) ,"accuracy_training": round((accuracy_training), 3)})

    else:
    # Retorna a previsão
        return jsonify({"accuracy_test": round((accuracy_test), 3) ,"accuracy_training": round((accuracy_training), 3)})





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

    # Recebe os dados enviados pelo formulário
    classifier_type = request.form["regression"]
    print(classifier_type)
    csv_file = request.files["csv_file"] 
    separator = request.form.get('separator')
    posicao = request.form.get('posicao')
    print(f"A posicao eh {posicao}")

    deployBoolean = request.form.get('deployBoolean')
   
    

    # Lê o arquivo CSV
    csv_data = pd.read_csv(io.BytesIO(csv_file.read()),sep = separator, encoding = 'utf-8')
    csv_tranform = csv_data.copy()
    print(csv_data)


    if deployBoolean == "true":
        csv_file_deploy = request.files["csv_deploy"]
        csv_deploy = pd.read_csv(io.BytesIO(csv_file_deploy.read()), sep= separator, encoding='utf-8')
        print(f"-----------------------------------------------------")
        csv_deploy = csv_deploy.to_numpy().tolist()
        print(csv_deploy)
        print(f"-----------------------------------------------------")
    
    
    
    #Vetores necessarios para diferentes tipos de regressão e diferentes x e y
    name_Regression_fit = ["simple_linear_regression", "multiple_linear_regression", "regression_with_decision_tree", "regression_with_random_forest","regression_with_xgboost","regression_with_light_gbm", "regression_with_catboost"]
    name_Regression_poly = ["polynomial_regression"]
    name_Regression_SVR = ["regression_by_support_vectors"]



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
    
    
    # def is_not_nan(value):
    #     try:
    #         # Tenta converter o valor para float
    #         float_value = float(value)
    #     except ValueError:
    #         # Se a conversão falhar, não é um número válido
    #         return False
    #     # Verifica se o valor convertido é NaN
    #     return not math.isnan(float_value)
    
    
    # def features_forms():
    #      # Recebe as características do formulário    
    #     features = []
    #     if (classifier_type != "simple_linear_regression" and classifier_type != "polynomial_regression"):
    #         for i in range(1, X.shape[1] + 1):
    #             form_value = request.form[f"feature{i}"]
    #             if is_not_nan(form_value):
    #                 feature = float(form_value)
    #                 print(f"feature if1 {feature}")
    #             else:
    #                 feature = form_value
    #                 print(f"feature else1 {feature}")
    #             features.append(feature)
            
    #     else:
    #         form_value = request.form[posicao]
    #         if is_not_nan(form_value):
    #                 feature = float(form_value)
    #                 print(f"feature if2 {feature}")
    #         else:
    #             feature = form_value
    #             print(f"feature else2 {feature}")
    #         features.append(feature)
    #     return features
        
    

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
        y =  csv_data.iloc[:, 3].values
        return X, y

    def all_columns():    
        # Separa as características e o alvo
        X = csv_data.iloc[:, :-1]
        y = csv_data.iloc[:, -1]
        print(X, y)
        return X, y
    
 
    if(classifier_type == "simple_linear_regression"):
        reg =  LinearRegression()
        X,y = some_columns()
        
    elif(classifier_type == "multiple_linear_regression"):
        reg =  LinearRegression()
        X,y = all_columns()
    elif(classifier_type == "polynomial_regression"):
        reg =  LinearRegression()
        X,y = some_columns()

    elif(classifier_type == "regression_by_support_vectors"):
        parameters = parameters_forms()
        reg =  SVR(kernel= parameters[0])
        X,y = all_columns()
    
    elif(classifier_type == "regression_with_decision_tree"):
        parameters = parameters_forms()
        reg =  DecisionTreeRegressor(max_depth= int(parameters[0]), random_state=int(parameters[1]))
        X,y = all_columns()
    
    elif(classifier_type == "regression_with_random_forest"):
        parameters = parameters_forms()
        reg = RandomForestRegressor(n_estimators=int(parameters[0]), criterion=parameters[1], max_depth=int(parameters[2]), random_state = int(parameters[3])) 
        X,y = all_columns()

    elif(classifier_type == "regression_with_xgboost"):
        parameters = parameters_forms()
        reg = XGBRegressor(n_estimators=int(parameters[0]), max_depth=int(parameters[1]), learning_rate=float(parameters[2]), objective=parameters[3], random_state=int(parameters[4]))
        X,y = all_columns()

    elif(classifier_type == "regression_with_light_gbm"):
        parameters = parameters_forms()
        reg = lgbm.LGBMRegressor(num_leaves=int(parameters[0]), max_depth=int(parameters[1]), learning_rate=float(parameters[2]), n_estimators=int(parameters[3]), random_state=int(parameters[4]))
        X,y = all_columns()
    
    elif(classifier_type == "regression_with_catboost"):
        parameters = parameters_forms()
        reg = CatBoostRegressor(iterations=int(parameters[0]), learning_rate=float(parameters[1]), depth = int(parameters[2]), random_state = int(parameters[3]))
        X,y = all_columns()

    
    #Base treino e teste 
    x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.3, random_state = 0)
    
    
    
    if classifier_type in name_Regression_fit:
        determinationCoefficientTraining,determinationCoefficientTest,absolute,MeanSquaredError = train_fit(reg,x_teste, x_treino, y_teste, y_treino)
        

    elif classifier_type in name_Regression_poly:
        determinationCoefficientTraining,determinationCoefficientTest,absolute,MeanSquaredError = train_poly(reg,x_teste, x_treino, y_teste, y_treino)
    
    elif classifier_type in name_Regression_SVR:
        determinationCoefficientTraining,determinationCoefficientTest,absolute,MeanSquaredError = train_SVR(reg,x_teste, x_treino, y_teste, y_treino)
    
#     features = features_forms()
#     print(f"features final : {features}")
#    #prediction = reg.predict([features])

#     cont_coluns = 0
#     num_features = len(features)  # Armazena o comprimento da lista de features

#     for coluna in csv_tranform.columns:
#         # Verifica se o tipo da coluna é 'object' e se todos os elementos não são números
#         if csv_tranform[coluna].dtype == 'object' and not csv_tranform[coluna].apply(lambda x: isinstance(x, (int, float))).all():
#             # Verifica se o índice está dentro do alcance da lista de features
#             if cont_coluns < num_features:
#                 # Codifica a feature usando o dicionário
#                 features[cont_coluns] = meu_dicionarioencoder[coluna][meu_dicionario[coluna].index(features[cont_coluns])]
#         cont_coluns += 1

    
    




    
    
    
    # print(f"features: {features}")
    if deployBoolean == "true":
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
        return jsonify({"prediction":val,"determinationCoefficientTraining": round((determinationCoefficientTraining *100), 3), "determinationCoefficientTest": round((determinationCoefficientTest*100),3), "abs":round((absolute), 3), 
                    "MeanSquaredError":round((MeanSquaredError), 3)})
    else:    
        # Retorna os valores dos testes
        return jsonify({"determinationCoefficientTraining": round((determinationCoefficientTraining *100), 3), "determinationCoefficientTest": round((determinationCoefficientTest*100),3), "abs":round((absolute), 3), 
                    "MeanSquaredError":round((MeanSquaredError), 3)})



@app.route('/graphicAnalysisPost', methods=['POST'])
def submit_selections_graphicAnalysis():
    typeGraphic = request.form.get('typeGraphic')
    print(f"typeGraphic {typeGraphic}")
    csv_file = request.files.get('csvFile')
    selections = request.form.get('selections')
    selections = json.loads(selections)
    print(selections)
    # Processar o CSV
    separator = ","
    csv_data = pd.read_csv(io.BytesIO(csv_file.read()), sep=separator, encoding='utf-8')
    print(csv_data)
    
    
    
    def createGraph(key, values, data):
        
        if values == "histogramas":
            graf = go.Figure(data=[go.Histogram(x=csv_data[key], nbinsx=60)])
            graf.update_layout(width=800, height=500, title_text='Distribuição por '+ key, paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = graf.to_json()
            data[key] = graf_json

        elif values == "gráficos de pizza":
            counts = csv_data[key].value_counts()
            print(f"counts: {counts}\ncounts.index: {counts.index}\ncounts.values: {counts.values}")
            graf = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values)])
            graf.update_layout(width=800, height=500, title_text='Distribuição por '+ key, paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = graf.to_json()
            data[key] = graf_json


        elif values == "gráficos de linha":
            counts = csv_data[key].value_counts()
            graf = go.Figure(data=[go.Scatter(x=counts.index, y=counts.values, mode='lines+markers', name='Distribuição por ' + key)])
            graf.update_layout(width=800, height=500, title_text='Distribuição por ' + key,paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = graf.to_json()
            data[key] = graf_json
        
        elif values == "gráficos de barras":
            counts = csv_data[key].value_counts()
            graf = go.Figure(data=[go.Bar(x=counts.index,y=counts.values,name='Distribuição por ' + key)])
            graf.update_layout(title='Distribuição por ' + key, xaxis_title='Categoria', yaxis_title='Frequência', width=800, height=500,paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = graf.to_json()
            data[key] = graf_json
        
        elif values == "gráficos de dispersão":
            counts = csv_data[key].value_counts()
            graf = go.Figure(data=[go.Scatter(x=counts.index, y=counts.values, mode='markers', name='Distribuição por ' + key)])
            graf.update_layout(width=800, height=500, title='Distribuição por ' + key, xaxis_title='X',yaxis_title='Y', paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = graf.to_json()
            data[key] = graf_json
        
        elif values == "boxplot":
            counts = csv_data[key].value_counts()
            graf = go.Figure(data=[go.Box( y= counts.values ,name='Distribuição por ' + key)])
            graf.update_layout(width=800, height=500,  yaxis_title='Valores', title='Distribuição por ' + key, paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
            graf_json = graf.to_json()
            data[key] = graf_json
        return data
    

    def createGraph2Var(graphicType, var1, var2, data):
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
        
        elif(graphicType == "Gráficos de Barras"):
            hist_fig = px.bar(df_grouped, x= var1, y='soma', color=var2,title=f'Gráficos de barras de {var1} por {var2}', barmode='stack',color_discrete_map=color_discrete_map, width=800, height=500)
            hist_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = hist_fig.to_json()
            data[graphicType] = graf_json
        
        elif(graphicType == "Gráficos de Linha"):
            hist_fig = px.line(df_grouped, x= var1, y='soma', color=var2,title=f'Gráficos de linha de {var1} por {var2}',color_discrete_map=color_discrete_map, width=800, height=500)
            hist_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = hist_fig.to_json()
            data[graphicType] = graf_json

        elif(graphicType == "Gráficos de Pizza"):
            hist_fig = px.pie(df_grouped,names=var1,values='soma', color=var2,title=f'Gráficos de pizza de {var1} por {var2}', color_discrete_map=color_discrete_map, width=800, height=500)
            hist_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = hist_fig.to_json()
            data[graphicType] = graf_json
        
        elif(graphicType == "Gráficos de Dispersão"):
            hist_fig = px.scatter(df_grouped,x= var1, y='soma',color=var2,title=f'Gráficos de dispersão de {var1} por {var2}', color_discrete_map=color_discrete_map, width=800, height=500)
            hist_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)')
            graf_json = hist_fig.to_json()
            data[graphicType] = graf_json

        return data
    
    
    
    def typeGraph(selections):
        data = dict()
        for key, values in selections.items():
          print(f"Chave: {key}\nValor: {values}\n")
          if(typeGraphic == "true"):
            data = createGraph(key, values, data)  
        return data
    
    def typeGraph2Var(selections):
        data = dict()
        graphicType = list(selections.values())[0]
        var1 = list(selections.values())[1]
        var2 = list(selections.values())[2]
        data = createGraph2Var(graphicType, var1, var2, data)
        print(f"Data {data}")
        return data
    
    
    if(typeGraphic == "true"):
        data = typeGraph(selections)
        print(f" Tipo : {type(data)}")
        return jsonify(data)

    elif(typeGraphic == "false"):
        data = typeGraph2Var(selections)
        print(f" Tipo : {type(data)}")
        return jsonify(data)



@app.route('/associationRulesPost', methods=['POST'])
def submit_selections_associationRules():
    typeGraphic = request.form.get('typeGraphic')
    print(f"typeGraphic {typeGraphic}")
    csv_file = request.files.get('csvFile')
    selections = request.form.get('selections')
    selections = json.loads(selections)
    print(selections)
    # Processar o CSV
    separator = ","
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
    return jsonify({"dados": result})


   



if __name__ == "__main__":
    app.run(debug=True)
