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
# Importando a biblioteca do XGBOOST (pip install xgboost) ou no colab(!pip install xgboost)
from xgboost import XGBClassifier
# Importando a biblioteca do LIGHTGBM (!pip install lightgbm) ou no colab(!pip install lightgbm)
import lightgbm as lgbm
# Importando a biblioteca do CATBOOST (!pip install catboost) ou no colab(!pip install catboost)
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


# Regressão
 
# Análise da Normalidade
import scipy.stats as stats

#Teste Lilliefors (Kolmogorov_Sminorv) Ho = distribuição normal : p > 0.05 Ha = distribuição != normal : p <= 0.05 (pip install statsmodels)
import statsmodels
from statsmodels.stats.diagnostic import lilliefors




app = Flask(__name__)





# Rota para o frontend
@app.route("/")
def index():
    return render_template("classifiers.html")

@app.route("/typedata")
def typedata():
    return render_template("typedata.html")



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

    def is_not_nan(value):
        return isinstance(value, (int, float)) and not math.isnan(value)

    
    def is_not_nan(value):
        try:
            # Tenta converter o valor para float
            float_value = float(value)
        except ValueError:
            # Se a conversão falhar, não é um número válido
            return False
        # Verifica se o valor convertido é NaN
        return not math.isnan(float_value)

    # Recebe as características do formulário
    features = []
    for i in range(1, X.shape[1] + 1):
        form_value = request.form[f"feature{i}"]
        if is_not_nan(form_value):
            feature = float(form_value)
            print(f"feature if {feature}")
        else:
            feature = form_value
            print(f"feature else {feature}")
        features.append(feature)

    print(f"features aaaaa: {features}")
    print(csv_data)
    print(csv_tranform)

    cont_coluns = 0
    num_features = len(features)  # Armazena o comprimento da lista de features

    for coluna in csv_tranform.columns:
        # Verifica se o tipo da coluna é 'object' e se todos os elementos não são números
        if csv_tranform[coluna].dtype == 'object' and not csv_tranform[coluna].apply(lambda x: isinstance(x, (int, float))).all():
            # Verifica se o índice está dentro do alcance da lista de features
            if cont_coluns < num_features:
                # Codifica a feature usando o dicionário
                features[cont_coluns] = meu_dicionarioencoder[coluna][meu_dicionario[coluna].index(features[cont_coluns])]
        cont_coluns += 1

    
    
    
    
    print(f"features: {features}")
    prediction = clf.predict([features])
    
    

    
    if(csv_tranform.iloc[:, -1].dtype == 'object'):
        # Convertendo as chaves do dicionário em uma lista
        lista_chaves = list(meu_dicionarioencoder.keys())

        # Obtendo a última chave
        ultima_chave = lista_chaves[-1]

        # Encontrando o índice da previsão na lista correspondente à última chave
        indice_predicao = meu_dicionarioencoder[ultima_chave].index(prediction[0])

        val = meu_dicionario[ultima_chave][indice_predicao]
    
    else: 
        val = prediction[0]
    




    # Retorna a previsão
    return jsonify({"prediction": f"{val}", "accuracy_test": accuracy_test, "accuracy_training":accuracy_training })





# Rota para fazer testes de normalidade e correlação para regressão linear simples
@app.route("/typedatatest", methods=["POST"])
def typedatatest():
    csv_file = request.files["csv_file"]
    separator = request.form.get('separator')
    # Lê o arquivo CSV
    csv_data = pd.read_csv(io.BytesIO(csv_file.read()),sep = separator, encoding = 'utf-8')
    print(csv_data)

    #Teste Lilliefors (Kolmogorov_Sminorv) Ho = distribuição normal : p > 0.05 Ha = distribuição != normal : p <= 0.05
    def TesteLilliefors():
        continuesNormality = False
        for coluns in csv_data.columns:
            estatistica, p = statsmodels.stats.diagnostic.lilliefors(csv_data[coluns], dist = 'norm')
            if(p > 0.05):
                continuesNormality = True
            print('Estatistica de teste: {}'.format(estatistica))
            print('p-valor: {}'.format(p))
        return continuesNormality
    
    continuesNormality = TesteLilliefors()

    
    def CorrelaçãoLinear(): # Pearson (distribuição normal), Spearman (distribuição não normal), Kendall (distribuição não normal com quantidade pequena de amostras < 30)
        #Ho = não há corrrelação linear: p > 0,05, Ha = existe correlação linear: p <= 0,05
        if continuesNormality == True:
            #Pearson
            correlation = csv_data.corr(method='pearson')
            print(correlation)
        elif  continuesNormality == False:
            correlation = csv_data.corr(method='spearman')
        
        # Converter para JSON com orient='columns' para manter os cabeçalhos das colunas
        json_correlation = correlation.to_json(orient='columns')
        print(json_correlation)

        return json_correlation    

    
    json_correlation = CorrelaçãoLinear()    


    # Retorna os valores dos testes
    return jsonify({"linearCorrelation":json_correlation, "continuesNormality":continuesNormality})


if __name__ == "__main__":
    app.run(debug=True)
