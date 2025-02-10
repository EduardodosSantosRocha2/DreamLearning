let spinner = document.getElementById('spinner');
spinner.style.display = 'none';
var filename;
var parametersColection;
var datalistOptions;
var radioButtons = document.querySelectorAll('input[type="radio"]')
const valCheckBox = {
    "Floresta Aleatória": ["n_estimator", "criterion", "random_state", "max_depht"],
    "SVM": ["kernel", "random_state", "C"],
    "Regressão Logistica": ["random_state", "max_iter", "penalty", "tol", "C", "solver"],
    "KNN": ["n_neighbors", "metric"],
    "Arvore de Decisão": ["criterion", "random_state", "max_depth"],
    "XGBOOST": ["max_depth", "learning_rate", "n_estimators", "objective", "random_state"],
    "LIGHTGBM": ["num_leaves", "objective", "max_depth", "learning_rate", "max_bin", "num_boost_round"],
    "CATBOOST": ["task_type", "iterations", "learning_rate", "depth", "random_state", "eval_metric"]
}

let values = Object.keys(valCheckBox);




function stars() {
    const count = 500;
    const scene = document.querySelector(".scene");
    const fragment = document.createDocumentFragment();
    const width = window.innerWidth;
    const height = window.innerHeight;

    for (let i = 0; i < count; i++) {
        const star = document.createElement("i");
        star.className = "istars";
        const x = Math.random() * width;
        const y = Math.random() * height;
        const duration = 5 + Math.random() * 10;
        const size = 1 + Math.random() * 2;

        star.style.cssText = `
            left: ${x}px;
            top: ${y}px;
            width: ${size}px;
            height: ${size}px;
            animation-duration: ${duration}s;
            animation-delay: ${Math.random() * duration}s;
        `;

        fragment.appendChild(star);
    }

    scene.appendChild(fragment);
}

stars();


function toggleCode() {
    const codeBlock = document.getElementById('codeBlock');
    codeBlock.style.display = codeBlock.style.display === 'block' ? 'none' : 'block';
}

function copyCode(event) {
    const code = document.querySelector('#codeBlock code').innerText;
    navigator.clipboard.writeText(code).then(() => {
        showNotification(event.pageY, event.pageX, 'Copiado!');
    }, () => {
        showNotification(event.pageY, event.pageX, 'Falha ao copiar o código.');
    });
}

function showNotification(top, left, message = 'Código copiado!') {
    const notification = document.getElementById('notification');
    notification.querySelector('p').innerText = message;
    notification.style.display = 'block';
    notification.style.top = `${top - 50}px`; // Ajuste a posição vertical conforme necessário
    notification.style.left = `${left - 100}px`; // Ajuste a posição horizontal conforme necessário
    setTimeout(() => {
        notification.style.display = 'none';
    }, 1500); // O balão ficará visível por 1.5 segundos
}



document.addEventListener("DOMContentLoaded", function () {

    var separator = "";
    var deployBoolean = "";



    const radioButtons = document.querySelectorAll('input[name="answer-dark"]');
    const deployDiv = document.querySelector('.deploy');

    function toggleDeployDiv() {
        const selectedValue = document.querySelector('input[name="answer-dark"]:checked').value;
        console.log("Escolha: " + selectedValue)
        if (selectedValue === 'yes') {
            deployDiv.style.display = 'block';
            deployBoolean = "true";
        } else {
            deployDiv.style.display = 'none';
            deployBoolean = "false";
        }
    }

    // Adiciona o event listener para cada rádio button
    radioButtons.forEach(radio => {
        radio.addEventListener('change', toggleDeployDiv);
    });

    // Verifica o estado inicial dos rádios
    toggleDeployDiv();


    document
        .getElementById("classifier")
        .addEventListener("change", function (event) {
            var select = document.querySelector("#classifier");
            var option = select.children[select.selectedIndex];
            var textClassifier = option.textContent;

            if (textClassifier === "Random Forest") {
                parametersColection = {
                    n_estimators: "number",
                    criterion: "text",
                    random_state: "number",
                    max_depth: "number",
                };

                datalistOptions = {
                    n_estimators: [100, 150, 200, 250],
                    criterion: ["gini", "entropy", "log_loss"],
                    random_state: [0, 1, 2, 3, 4, 5],
                    max_depth: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                };

                createParameters(4, parametersColection, datalistOptions);

            }

            if (textClassifier === "SVM") {
                console.log("chegou");
                parametersColection = {
                    kernel: "text",
                    random_state: "number",
                    C: "number",
                };

                datalistOptions = {
                    kernel: ["linear", "poly", "rbf", "sigmoid", "precomputed"],
                    random_state: [0, 1, 2, 3, 4, 5],
                    C: [1, 2, 5, 10]
                };

                createParameters(3, parametersColection, datalistOptions);
            }

            if (textClassifier === "LOGISTICS REGRESSION") {
                console.log("chegou");
                parametersColection = {
                    random_state: "number",
                    max_iter: "number",
                    penalty: "text",
                    tol: "number",
                    C: "number",
                    solver: "text",
                };

                datalistOptions = {
                    random_state: [0, 1, 2, 3, 4, 5],
                    max_iter: [100, 200, 300, 400, 500],
                    penalty: ["l1", "l2"],
                    tol: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                    C: [1, 2, 5, 10],
                    solver: ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
                };

                createParameters(6, parametersColection, datalistOptions);
            }

            if (textClassifier === "KNN") {
                console.log("chegou");
                parametersColection = {
                    n_neighbors: "number",
                    metric: "text",
                };

                datalistOptions = {
                    n_neighbors: [1, 3, 5, 7, 9],
                    metric: ["minkowski", "euclidean", "manhattan", "chebyshev", "hamming", "cosine"]
                };

                createParameters(2, parametersColection, datalistOptions);
            }

            if (textClassifier === "DECISION TREE") {
                console.log("chegou");
                parametersColection = {
                    criterion: "text",
                    random_state: "number",
                    max_depth: "text",
                };

                datalistOptions = {
                    criterion: ["gini", "entropy", "log_loss"],
                    random_state: [0, 1, 2, 3, 4, 5],
                    max_depth: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                };

                createParameters(3, parametersColection, datalistOptions);
            }

            if (textClassifier === "XGBOOST") {
                console.log("chegou");
                parametersColection = {
                    max_depth: "number",
                    learning_rate: "number",
                    n_estimators: "number",
                    objective: "text",
                    random_state: "number",
                };

                datalistOptions = {
                    max_depth: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    learning_rate: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                    n_estimators: [100, 150, 200, 250],
                    objective: ["binary:logistic", "binary:logitraw", "binary:hinge", "count:poisson", "survival:cox", "survival:aft", "multi:softmax", "multi:softprob", "rank:ndcg", "rank:map", "rank:pairwise"],
                    random_state: [0, 1, 2, 3, 4, 5]
                };

                createParameters(5, parametersColection, datalistOptions);
            }

            if (textClassifier === "LIGHTGBM") {
                console.log("chegou");
                parametersColection = {
                    num_leaves: "number",
                    objective: "text",
                    max_depth: "number",
                    learning_rate: "number",
                    max_bin: "number",
                    num_boost_round: "number",
                };

                datalistOptions = {
                    num_leaves: [100, 150, 200, 250],
                    objective: ["binary", "multiclass", "multiclassova", "ova", "xentropy", "xentlambda"],
                    max_depth: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    learning_rate: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                    max_bin: [100, 150, 200, 250],
                    num_boost_round: [100, 150, 200, 250],
                };

                createParameters(6, parametersColection, datalistOptions);
            }

            if (textClassifier === "CATBOOST") {
                console.log("chegou");
                parametersColection = {
                    task_type: "text",
                    iterations: "number",
                    learning_rate: "number",
                    depth: "number",
                    random_state: "number",
                    eval_metric: "text",
                };

                datalistOptions = {
                    task_type: ["CPU", "GPU"],
                    iterations: [100, 150, 200, 250],
                    learning_rate: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                    depth: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    random_state: [0, 1, 2, 3, 4, 5],
                    eval_metric: ["Accuracy", "Logloss", "AUC"],
                };

                createParameters(6, parametersColection, datalistOptions);
            }
        });



    function createParameters(size, parametersColection, datalistOptions) {
        const datalists = document.querySelectorAll('datalist');
        datalists.forEach(datalist => datalist.remove());

        const parametersDiv = document.getElementById("parameters");
        parametersDiv.innerHTML = "";

        var keys = Object.keys(parametersColection);
        for (var i = 0; i < size; i++) {
            console.log(keys[i]);
            console.log(parametersColection[keys[i]]);
            const parametersLabel = document.createElement("label");
            parametersLabel.textContent = keys[i] + ":";
            parametersDiv.appendChild(parametersLabel);
            const parametersInput = document.createElement("input");
            parametersInput.type = parametersColection[keys[i]];
            parametersInput.step = "0.0000000001";
            parametersInput.name = "parameters" + (i + 1);
            parametersInput.className = "nes-input";

            if (datalistOptions[keys[i]]) {
                const datalistID = `datalist-${keys[i]}`;
                parametersInput.setAttribute("list", datalistID);

                const dataList = document.createElement("datalist");
                dataList.id = datalistID;

                datalistOptions[keys[i]].forEach((optionValue) => {
                    const option = document.createElement("option");
                    option.value = optionValue;
                    dataList.appendChild(option);
                })

                parametersDiv.appendChild(dataList);
            }


            parametersDiv.appendChild(parametersInput);
            parametersDiv.appendChild(document.createElement("br"));

        }
    }

    document
        .getElementById("csv_file")
        .addEventListener("change", function (event) {
            console.log("Evento de mudança detectado");
            const file = event.target.files[0];
            if (!file) {
                console.error("Nenhum arquivo selecionado.");
                return;
            }
            console.log("Arquivo selecionado: ", file);

            const reader = new FileReader();

            reader.onload = function (e) {
                const text = e.target.result;
                if (!text) {
                    console.error("Falha ao ler o arquivo.");
                    return;
                }
                console.log("Conteúdo do arquivo: ", text);

                const lines = text.split("\n");
                if (lines.length === 0) {
                    console.error("Arquivo CSV vazio ou inválido.");
                    return;
                }

                // Verifica se a primeira linha contém uma vírgula
                if (lines[0].includes(",")) {
                    separator = ",";
                }
                // Se não, verifica se contém um ponto e vírgula
                else if (lines[0].includes(";")) {
                    separator = ";";
                } else {
                    console.error("Separador não reconhecido.");
                    return;
                }

                let header = lines[0].split(separator);
                if (lines.length < 2) {
                    console.error("Arquivo CSV não contém dados suficientes.");
                    return;
                }
                //     let text_or_number = lines[1].split(separator);

                //     console.log("Cabeçalho: ", header);
                //     const featuresDiv = document.getElementById("features");
                //     featuresDiv.innerHTML = "";

                //     for (let i = 0; i < header.length - 1; i++) {
                //         const featureLabel = document.createElement("label");
                //         featureLabel.textContent = header[i] + ":";
                //         featuresDiv.appendChild(featureLabel);
                //         const featureInput = document.createElement("input");
                //         if (!isNaN(text_or_number[i])) {
                //             featureInput.type = "number";
                //             featureInput.step = "0.0000000001";
                //         } else {
                //             featureInput.type = "text";
                //         }
                //         featureInput.name = "feature" + (i + 1);
                //         featuresDiv.appendChild(featureInput);
                //         featuresDiv.appendChild(document.createElement("br"));
                //     }
                // };

                // reader.onerror = function (e) {
                //     console.error("Erro ao ler o arquivo: ", e);
            };
            filename = file.name.split(".csv")[0];
            reader.readAsText(file);
        });




    document
        .getElementById("prediction-form")
        .addEventListener("submit", function (event) {
            spinner.style.display = 'block';
            document.getElementById("result").innerText = "";
            event.preventDefault();
            const formData = new FormData(this);
            formData.append("separator", separator);
            formData.append("deployBoolean", deployBoolean);





            fetch("/predict", {
                method: "POST",
                body: formData,
            })
                .then((response) => {
                    if (!response.ok) {
                        throw new Error("Erro ao fazer a solicitação.");
                    }
                    return response.json();
                })
                .then((data) => {
                    document.getElementById(
                        "result"
                    ).innerText = "";
                    spinner.style.display = 'none';
                    let resultText = `Acuracia teste: ${data.accuracy_test}%<br>`;
                    resultText += `Acuracia treino: ${data.accuracy_training}%<br>`;

                    // Verifica se data.prediction não é vazio antes de adicionar ao texto resultante
                    if (data.prediction !== undefined && data.prediction !== null) {
                        resultText = `Previsão: ${data.prediction}<br>` + resultText;
                    }

                    document.getElementById("result").innerHTML = `<div class="preformatted-text">${"<p>" + resultText + "</p>"}</div>`;
                    document.getElementById("windowcode").innerHTML = data.code;





                    const classifier = formData.get('classifier');
                    const parameters = {};
                    formData.forEach((value, key) => {
                        if (key.startsWith('parameters')) { // Filtra apenas as chaves que começam com 'parameters'
                            parameters[key] = value; // Adiciona ao objeto
                        }
                    });
                    const today = new Date();
                    const dataAtual = today.toISOString().split('T')[0];



                    saveTransaction(classifier, "0", filename, parameters, data.accuracy_test, data.accuracy_training, dataAtual)


                })
                .catch((error) => {
                    console.error("Erro:", error);
                    document.getElementById(
                        "result"
                    ).innerText = `Erro: ${error.message}`;
                });
        });
});





function saveTransaction(Algorithm, CrossValidation, NameDatabase, Parameters, TestAccuracy, TrainingAccuracy, date) {

    const result = {};
    const values = Object.values(Parameters);
    Object.keys(parametersColection).forEach((key, index) => {
        result[key] = values[index];
    });



    // Criar a transação
    const transaction = {
        Algorithm: Algorithm,
        CrossValidation: CrossValidation,
        NameDatabase: NameDatabase,
        Parameters: result,
        TestAccuracy: TestAccuracy,
        TrainingAccuracy: TrainingAccuracy,
        date: date,
        type: "Classificação",
        user: {
            uid: firebase.auth().currentUser.uid
        }
    }

    // Adicionar ao Firestore
    firebase.firestore()
        .collection('transactions')
        .add(transaction)
        .then(() => {
            console.log("Cadastrado no DBA");
        })
        .catch(error => {
            alert("Erro ao salvar transação!");
        })
}


radioButtons.forEach(radio => {
    radio.addEventListener("change", () => {
        gerarCheckBox(radio.value);
    });
});


function gerarCheckBox(typeAlg) {
    const chat = document.getElementById("checkboxs");
    checkboxs.innerHTML = "";
    key = values
    parametersAlg = valCheckBox[typeAlg]

    parametersAlg.forEach(val => {


        const div = document.createElement('div');
        div.className = "form-check form-check-inline"

        const input = document.createElement('input');
        input.type = "checkbox";
        input.id = val;
        input.name = val;
        input.value = typeAlg;
        input.className = "form-check-input";

        const label = document.createElement('label');
        label.className = "form-check-label";
        label.htmlFor = val;

        label.textContent = val;

        div.appendChild(input);
        div.appendChild(label);

        chat.appendChild(div);


    });
}



