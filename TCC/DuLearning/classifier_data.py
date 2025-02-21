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


class classifierData:
    
    def encode_columns(self, csv_data):
        """
        (PT-BR)

        Codifica as colunas do DataFrame do tipo 'object' usando LabelEncoder.
        Retorna o DataFrame modificado, um dicionário com os valores únicos antes da codificação
        e outro com os valores únicos depois da codificação.
        
        (EN)

        Encodes the columns of a DataFrame of type 'object' using LabelEncoder.
        Returns the modified DataFrame, a dictionary with the unique values ​​before encoding
        and another with the unique values ​​after encoding.
        """
        meu_dicionario = {}
        meu_dicionarioencoder = {}

        encoder = LabelEncoder()

        

        """
        (PT-BR)
        Itere sobre todas as colunas do DataFrame
        
        (EN)
        Iterate over all columns of the DataFrame

        """

        for coluna in csv_data.columns:
            # (PT-BR) Verifique se a coluna é do tipo 'object' e não é numérica  (EN) Check if the column is of type 'object' and not numeric
            if csv_data[coluna].dtype == 'object' and not csv_data[coluna].apply(lambda x: isinstance(x, (int, float))).all():            
                meu_dicionario[coluna] = csv_data[coluna].unique().tolist() 
                csv_data[coluna] = encoder.fit_transform(csv_data[coluna]) # (PT-BR) Ajuste o encoder aos dados e transforme a coluna (EN) Fit the encoder to the data and transform the column
                meu_dicionarioencoder[coluna] = csv_data[coluna].unique().tolist()

        return csv_data, meu_dicionario, meu_dicionarioencoder

    
    def train_test_split(self, csv_data):
        # (PT-BR) Separa as características e o alvo Separates (EN) the features and the target
        x = csv_data.iloc[:, :-1]
        y = csv_data.iloc[:, -1]

        # (PT-BR) Base treino e teste (EN) Training and testing base
        x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3, random_state = 0)

        return x_treino, x_teste, y_treino, y_teste
    

    def train(self,classifier_type, x_treino, x_teste, y_treino, y_teste, separator, deployBoolean, csv_tranform, csv_data, csv_deploy,meu_dicionario,meu_dicionarioencoder):
        # "Variavel da matriz de confusão treino e teste"
        confusionMatrixTest = "";
        confusionMatrixTraning = "";
        
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

        name_Classifier_fit = ["Random Forest", "SVM","LOGISTICS REGRESSION","KNN","DECISION TREE","XGBOOST"]
        name_Classifier_predict = ["LIGHTGBM"]
        
        code = f"""<span class="bib">import</span> numpy <span class="bib">as</span> np\n<span class="bib">import</span> pandas <span class="bib">as</span> pd\n<span>dataframe = pd.read_csv(</span><span class= "link">'Adicione o caminho para o seu csv',sep ='{separator}', encoding = 'utf-8'</span>)</span>\n<span class="bib">from</span> sklearn.preprocessing <span class="bib">import</span> LabelEncoder\n<span class="keyword">encoder = </span><span class="function">LabelEncoder()</span>\n<span class = "comment">#Itere sobre todas as colunas do DataFrame</span>\n<span class="for">for </span><span>coluna <span class = "columns">in</span> dataframe.<span class = "columns">columns</span>:</span>\n<span class = "comment">  #Verifica se o tipo da coluna é 'object' e se todos os elementos não são números</span>\n<span class="if">  if </span><span>dataframe[coluna].dtype == <span class= "object">'object'</span></span><span class= "columns">and not</span> dataframe[coluna].apply(<span class = "columns">lambda</span> x: <span class = "instance">isinstance</span>(x, (<span class = "intfloat">int, float</span>))).<span class = "instance">all()</span></span>:</span>\n<span class = "comment">    #Ajuste o encoder aos dados e transforme a coluna</span>\n   <span>dataframe[coluna] = encoder.</span><span class="function">fit_transform<span>(dataframe[coluna])</span></span>\n\n<span class="keyword">previsores</span><span> = dataframe.iloc[:, :-1]</span>\n<span class="keyword">alvo</span><span> = dataframe.iloc[:, -1]</span>\n<span class="bib">from</span> sklearn.model_selection <span class="bib">import</span> train_test_split\n<span class="keyword">x_treino, x_teste, y_treino, y_teste =</span> <span class="function">train_test_split(previsores,alvo, test_size = 0.3, random_state = 0)</span><span class="comment">#Base treino e teste</span>\n"""

        
        # Escolhe o classificador com base na escolha do usuário
        if classifier_type == "Random Forest":
            clf = RandomForestClassifier(n_estimators=int(parameters[0]), criterion=parameters[1],
                                        random_state=int(parameters[2]), max_depth=int(parameters[3]))
            code += f"""<span class="bib">from</span> sklearn.ensemble <span class="bib">import </span>RandomForestClassifier\n<span class="keyword">randomforest = <span class="function">RandomForestClassifier</span></span>(n_estimators = {parameters[0]}, criterion = '{parameters[1]}', random_state = {parameters[2]}, max_depth = {parameters[3]})</span>\n\n<span class = "comment">#Treina o modelo</span>\n<span class="object">randomforest</span><span>.fit</span><span>(x_treino, y_treino)</span>\n\n<span class= "comment">#Importando a biblioteca para avaliação do algoritimo </span>\n<span class="bib">from</span> sklearn.metrics <span class="bib">import </span>accuracy_score, confusion_matrix, classification_report<span class = "comment">#Importando a biblioteca para avaliação do algoritimo</span>\n\n<span class= "comment">#Avaliação do teste </span>\n<span class="keyword">previsao_teste = </span><span class="object">randomforest</span><span>.predict(x_teste)</span>\n\n<span class= "comment">#Acuracia do teste </span>\n<span class="keyword">accuracia_teste = </span> <span class="function">accuracy_score</span><span>(y_teste, previsao_teste)*100</span>\n<span class="function">print</span><span>(f"Acuracia:{{accuracia_teste:.5f}}%")</span>\n\n<span class= "comment">#Avaliação do treino</span>\n<span class="keyword">previsao_treino = </span><span class="object">randomforest</span><span>.predict(x_treino)</span>\n\n<span class= "comment">#Acuracia do treino </span>\n<span class="keyword">accuracia_treino = </span> <span class="function">accuracy_score</span><span>(y_treino, previsao_treino)*100</span>\n<span class="function">print</span><span>(f"Acuracia:{{accuracia_treino:.5f}}%")</span>\n"""
            
            
        elif classifier_type == "SVM":
            clf = SVC(kernel=parameters[0], random_state=int(parameters[1]), C=float(parameters[2]))
            code += f"""<span class="bib">from</span> sklearn.svm <span class="bib">import </span>SVC\n<span class="keyword">svc= <span class="function">SVC</span></span>(kernel='{parameters[0]}', random_state={parameters[1]}, C={parameters[2]})</span>\n\n<span class = "comment">#Treina o modelo</span>\n<span class="object">svc</span><span>.fit</span><span>(x_treino, y_treino)</span>\n\n<span class= "comment">#Importando a biblioteca para avaliação do algoritimo </span>\n<span class="bib">from</span> sklearn.metrics <span class="bib">import </span>accuracy_score, confusion_matrix, classification_report<span class = "comment">#Importando a biblioteca para avaliação do algoritimo</span>\n\n<span class= "comment">#Avaliação do teste </span>\n<span class="keyword">previsao_teste = </span><span class="object">svc</span><span>.predict(x_teste)</span>\n\n<span class= "comment">#Acuracia do teste </span>\n<span class="keyword">accuracia_teste = </span> <span class="function">accuracy_score</span><span>(y_teste, previsao_teste)*100</span>\n<span class="function">print</span><span>(f"Acuracia:{{accuracia_teste:.5f}}%")</span>\n\n<span class= "comment">#Avaliação do treino</span>\n<span class="keyword">previsao_treino = </span><span class="object">svc</span><span>.predict(x_treino)</span>\n\n<span class= "comment">#Acuracia do treino </span>\n<span class="keyword">accuracia_treino = </span> <span class="function">accuracy_score</span><span>(y_treino, previsao_treino)*100</span>\n<span class="function">print</span><span>(f"Acuracia:{{accuracia_treino:.5f}}%")</span>\n"""
            


        elif classifier_type == "LOGISTICS REGRESSION":
            clf = LogisticRegression(random_state=int(parameters[0]), max_iter=int(parameters[1]), penalty=parameters[2],
                                    tol=float(parameters[3]), C=int(parameters[4]), solver=parameters[5])
            code += f"""<span class="bib">from</span> sklearn.linear_model <span class="bib">import </span>LogisticRegression\n<span class="keyword">regressaologistica= <span class="function">LogisticRegression</span></span>(random_state={parameters[0]}, max_iter={parameters[1]}, penalty='{parameters[2]}',tol={parameters[3]}, C = {parameters[4]}, solver= '{parameters[5]}')</span>\n\n<span class = "comment">#Treina o modelo</span>\n<span class="object">regressaologistica</span><span>.fit</span><span>(x_treino, y_treino)</span>\n\n<span class= "comment">#Importando a biblioteca para avaliação do algoritimo </span>\n<span class="bib">from</span> sklearn.metrics <span class="bib">import </span>accuracy_score, confusion_matrix, classification_report<span class = "comment">#Importando a biblioteca para avaliação do algoritimo</span>\n\n<span class= "comment">#Avaliação do teste </span>\n<span class="keyword">previsao_teste = </span><span class="object">regressaologistica</span><span>.predict(x_teste)</span>\n\n<span class= "comment">#Acuracia do teste </span>\n<span class="keyword">accuracia_teste = </span> <span class="function">accuracy_score</span><span>(y_teste, previsao_teste)*100</span>\n<span class="function">print</span><span>(f"Acuracia:{{accuracia_teste:.5f}}%")</span>\n\n<span class= "comment">#Avaliação do treino</span>\n<span class="keyword">previsao_treino = </span><span class="object">regressaologistica</span><span>.predict(x_treino)</span>\n\n<span class= "comment">#Acuracia do treino </span>\n<span class="keyword">accuracia_treino = </span> <span class="function">accuracy_score</span><span>(y_treino, previsao_treino)*100</span>\n<span class="function">print</span><span>(f"Acuracia:{{accuracia_treino:.5f}}%")</span>\n"""
            
        
        elif classifier_type == "KNN":
            clf = KNeighborsClassifier(n_neighbors=int(parameters[0]), metric=parameters[1])#tirei o p
            code += f"""<span class="bib">from</span> sklearn.neighbors <span class="bib">import </span>KNeighborsClassifier\n<span class="keyword">knn= <span class="function">KNeighborsClassifier</span></span>(n_neighbors={parameters[0]}, metric='{parameters[1]}'</span>\n\n<span class = "comment">#Treina o modelo</span>\n<span class="object">knn</span><span>.fit</span><span>(x_treino, y_treino)</span>\n\n<span class= "comment">#Importando a biblioteca para avaliação do algoritimo </span>\n<span class="bib">from</span> sklearn.metrics <span class="bib">import </span>accuracy_score, confusion_matrix, classification_report<span class = "comment">#Importando a biblioteca para avaliação do algoritimo</span>\n\n<span class= "comment">#Avaliação do teste </span>\n<span class="keyword">previsao_teste = </span><span class="object">knn</span><span>.predict(x_teste)</span>\n\n<span class= "comment">#Acuracia do teste </span>\n<span class="keyword">accuracia_teste = </span> <span class="function">accuracy_score</span><span>(y_teste, previsao_teste)*100</span>\n<span class="function">print</span><span>(f"Acuracia:{{accuracia_teste:.5f}}%")</span>\n\n<span class= "comment">#Avaliação do treino</span>\n<span class="keyword">previsao_treino = </span><span class="object">knn</span><span>.predict(x_treino)</span>\n\n<span class= "comment">#Acuracia do treino </span>\n<span class="keyword">accuracia_treino = </span> <span class="function">accuracy_score</span><span>(y_treino, previsao_treino)*100</span>\n<span class="function">print</span><span>(f"Acuracia:{{accuracia_treino:.5f}}%")</span>\n"""

        
        
        elif classifier_type == "DECISION TREE":
            clf = DecisionTreeClassifier(criterion=parameters[0], random_state=int(parameters[1]),
                                        max_depth=int(parameters[2]))
            
            code += f"""<span class="bib">from</span> sklearn.tree <span class="bib">import </span>DecisionTreeClassifier\n<span class="keyword">arvore= <span class="function">DecisionTreeClassifier</span></span>(criterion = '{parameters[0]}', random_state = {parameters[1]}, max_depth = {parameters[2]})</span>\n\n<span class = "comment">#Treina o modelo</span>\n<span class="object">arvore</span><span>.fit</span><span>(x_treino, y_treino)</span>\n\n<span class= "comment">#Importando a biblioteca para avaliação do algoritimo </span>\n<span class="bib">from</span> sklearn.metrics <span class="bib">import </span>accuracy_score, confusion_matrix, classification_report<span class = "comment">#Importando a biblioteca para avaliação do algoritimo</span>\n\n<span class= "comment">#Avaliação do teste </span>\n<span class="keyword">previsao_teste = </span><span class="object">arvore</span><span>.predict(x_teste)</span>\n\n<span class= "comment">#Acuracia do teste </span>\n<span class="keyword">accuracia_teste = </span> <span class="function">accuracy_score</span><span>(y_teste, previsao_teste)*100</span>\n<span class="function">print</span><span>(f"Acuracia:{{accuracia_teste:.5f}}%")</span>\n\n<span class= "comment">#Avaliação do treino</span>\n<span class="keyword">previsao_treino = </span><span class="object">arvore</span><span>.predict(x_treino)</span>\n\n<span class= "comment">#Acuracia do treino </span>\n<span class="keyword">accuracia_treino = </span> <span class="function">accuracy_score</span><span>(y_treino, previsao_treino)*100</span>\n<span class="function">print</span><span>(f"Acuracia:{{accuracia_treino:.5f}}%")</span>\n"""
        
        elif classifier_type == "XGBOOST":
            clf = XGBClassifier(max_depth=int(parameters[0]), learning_rate=float(parameters[1]),n_estimators=int(parameters[2]), objective=parameters[3], random_state=int(parameters[4]))
            code += f"""<span class="bib">from</span> xgboost <span class="bib">import </span>XGBClassifier\n<span class="keyword">xgboost= <span class="function">XGBClassifier</span></span>(max_depth={parameters[0]}, learning_rate={parameters[1]},n_estimators={parameters[2]}, objective='{parameters[3]}', random_state={parameters[4]})</span>\n\n<span class = "comment">#Treina o modelo</span>\n<span class="object">xgboost</span><span>.fit</span><span>(x_treino, y_treino)</span>\n\n<span class= "comment">#Importando a biblioteca para avaliação do algoritimo </span>\n<span class="bib">from</span> sklearn.metrics <span class="bib">import </span>accuracy_score, confusion_matrix, classification_report<span class = "comment">#Importando a biblioteca para avaliação do algoritimo</span>\n\n<span class= "comment">#Avaliação do teste </span>\n<span class="keyword">previsao_teste = </span><span class="object">xgboost</span><span>.predict(x_teste)</span>\n\n<span class= "comment">#Acuracia do teste </span>\n<span class="keyword">accuracia_teste = </span> <span class="function">accuracy_score</span><span>(y_teste, previsao_teste)*100</span>\n<span class="function">print</span><span>(f"Acuracia:{{accuracia_teste:.5f}}%")</span>\n\n<span class= "comment">#Avaliação do treino</span>\n<span class="keyword">previsao_treino = </span><span class="object">xgboost</span><span>.predict(x_treino)</span>\n\n<span class= "comment">#Acuracia do treino </span>\n<span class="keyword">accuracia_treino = </span> <span class="function">accuracy_score</span><span>(y_treino, previsao_treino)*100</span>\n<span class="function">print</span><span>(f"Acuracia:{{accuracia_treino:.5f}}%")</span>\n"""

        
        elif classifier_type == "LIGHTGBM":
            # Configurando o dataset de treino
            dataset = lgbm.Dataset(x_treino, label=y_treino)
            dicionarioParametros = {'num_leaves': int(parameters[0]),  # número de folhas
                                    'objective': parameters[1],       # classificação Binária
                                    'max_depth': int(parameters[2]),
                                    'learning_rate': float(parameters[3]),
                                    'max_bin': int(parameters[4])}
            code += f"""\n<span class="pip">!pip install lightgbm<span class="comment">#Use esse codigo para instalar a biblioteca</span></span>\n\n<span class="bib">import</span> lightgbm <span class="bib">as </span>lgbm\n<span>dataset = lgbm.<span class="columns">Dataset</span><span>(x_treino, label=y_treino)</span></span>\n<span class="keyword">dicionarioParametros = <span>{{</span><span class = "link">'num_leaves'</span>:<span class = "intfloat">{parameters[0]}</span>,<span class = "link">'objective'</span>:<span class = "intfloat">'{parameters[1]}'</span>,<span class="link">'max_depth'</span>:<span class = "intfloat">{parameters[2]}</span>,<span class = "link">'learning_rate'</span>:<span class = "intfloat">{parameters[3]}</span>,<span class = "link">'max_bin'</span>:<span class = "intfloat">{parameters[4]}</span><span>}}</span></span>\n\n<span class="comment">#Treinamento do modelo</span>\n<span><span class="keyword">lightgbm</span> = lgbm.<spna class="function">train</span>(dicionarioParametros, dataset, num_boost_round={parameters[5]})</span>\n\n<span class="comment">#Importando a biblioteca para avaliação do algoritimo</span>\n<span class="bib">from</span> sklearn.metrics <span class="bib">import </span>accuracy_score, confusion_matrix, classification_report\n\n<span class="comment">#Avaliação do teste</span>\n<span><span class="keyword">previsao_teste</span><span> = </span><span class="link">lightgbm</span><span></span>.predict(x_teste)</span>\n\n<span class="comment">#tranformando as previsao_teste em 0 e 1</span>\n<span class="for">for</span><span> i in range(<span class="columns">len</span>(previsao_teste)):</span>\n<span><span class="if">  if</span> previsao_teste[i] >=0.5:</span>\n<span>   previsao_teste[i] = 1</span>\n<span class="else">  else:</span>\n<span>   previsao_teste[i]=0</span>\n\n<span class="comment">#Acuracia do teste</span>\n<span>accuracia_teste = <span class="function">accuracy_score</span>(y_teste, previsao_teste)*100</span>\n<span><span class="function">print</span><span>(f"Acuracia teste: {{(accuracia_teste):.5f}}%")</span></span>\n\n\n<span class="comment">#Avaliação do treino</span>\n<span><span class="keyword">previsao_treino</span><span> = </span><span class="link">lightgbm</span><span></span>.predict(x_treino)</span>\n\n<span class="comment">#tranformando as previsao_treino em 0 e 1</span>\n<span class="for">for</span><span> i in range(<span class="columns">len</span>(previsao_treino)):</span>\n<span><span class="if">  if</span> previsao_treino[i] >=0.5:</span>\n<span>   previsao_treino[i] = 1</span>\n<span class="else">  else:</span>\n<span>   previsao_treino[i]=0</span>\n\n<span class="comment">#Acuracia do treino</span>\n<span>accuracia_treino = <span class="function">accuracy_score</span>(y_treino, previsao_treino)*100</span>\n<span><span class="function">print</span><span>(f"Acuracia teste: {{(accuracia_treino):.5f}}%")</span></span>\n"""
            
            
            # infosPredict = """<span class="keyword">previsao_teste = </span><span class="object">arvore</span><span>.predict(x_teste)</span>\n<span class="keyword">previsao_treino = </span><span class="object">arvore</span><span>.predict(x_treino)</span>\n"""

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


        if classifier_type in name_Classifier_fit:
            forecast_test, forecast_training = self.train_fit(clf,x_teste, x_treino, y_teste, y_treino)
            

        elif classifier_type in name_Classifier_predict:
            forecast_test, forecast_training = self.train_predict(clf,x_teste, x_treino, y_teste, y_treino)
        
        accuracy_test, confusionMatrixTest =  self.print_accuracy_score_test(forecast_test, y_teste)
        accuracy_training, confusionMatrixTraning =  self.print_accuracy_score_traning(forecast_training,y_treino)


        if deployBoolean == "true":
            val = self.deployBooleanTrue(csv_data, csv_tranform, csv_deploy, meu_dicionario, meu_dicionarioencoder, clf)
            return jsonify({"prediction": f"{val}","accuracy_test": round((accuracy_test), 3) ,"accuracy_training": round((accuracy_training), 3), 
                            "confusionMatrixTraning":confusionMatrixTraning.tolist(), "confusionMatrixTest":confusionMatrixTest.tolist()})
        
        else:
            return jsonify({"accuracy_test": round((accuracy_test), 3) ,"accuracy_training": round((accuracy_training), 3),"code":code, 
                            "confusionMatrixTraning":confusionMatrixTraning.tolist(), "confusionMatrixTest":confusionMatrixTest.tolist()})




    def train_fit(self, clf,x_teste, x_treino, y_teste, y_treino):
        print("chegou na funçao train_fit")
        # Treina o modelo
        clf.fit(x_treino, y_treino)# Treina o modelo
        #Avaliação do algoritmo
        forecast_test = clf.predict(x_teste)#Avaliação do teste
        #Avaliação de treino
        forecast_training = clf.predict(x_treino)#Avaliação de treino
        return forecast_test, forecast_training

        
    def train_predict(self,clf,x_teste, x_treino, y_teste, y_treino):
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
        
    def print_accuracy_score_test(self,forecast_test, y_teste):
        accuracy_test = accuracy_score(y_teste, forecast_test)*100
        print(f"Acuracia: {(accuracy_test):.5f}%\n")
        print(confusion_matrix(y_teste, forecast_test))
        return accuracy_test, confusion_matrix(y_teste, forecast_test)

        

    def print_accuracy_score_traning(self,forecast_training,y_treino):
        accuracy_training = accuracy_score(y_treino, forecast_training)*100
        print(f"Acuracia treino: {(accuracy_training):.5f}%\n")
        print(confusion_matrix(y_treino, forecast_training))
        return accuracy_training,confusion_matrix(y_treino, forecast_training)
    
    def deployBooleanTrue(self, csv_data, csv_tranform, csv_deploy, meu_dicionario, meu_dicionarioencoder, clf):
        print(csv_data)
        print(csv_tranform)

        cont_coluns = 0
        deploy = csv_deploy
        print(csv_deploy)
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

            return  meu_dicionario[ultima_chave][indice_predicao]
    
        else: 
            return prediction.tolist()