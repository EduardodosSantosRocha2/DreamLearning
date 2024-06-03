

document.addEventListener("DOMContentLoaded", function() {
    
    var separator = "";
    document
        .getElementById("classifier")
        .addEventListener("change", function (event) {
            var select = document.querySelector("#classifier");
            var option = select.children[select.selectedIndex];
            var textClassifier = option.textContent;
            console.log(textClassifier);
            const parametersDiv = document.getElementById("parameters");
            parametersDiv.innerHTML = "";
            if (textClassifier === "Random Forest") {
                var parametersColection = {
                    n_estimators: "number",
                    criterion: "text",
                    random_state: "number",
                    max_depth: "number",
                };
                var keys = Object.keys(parametersColection);
                for (var i = 0; i < 4; i++) {
                    console.log(keys[i]);
                    console.log(parametersColection[keys[i]]);
                    const parametersLabel = document.createElement("label");
                    parametersLabel.textContent = keys[i] + ":";
                    parametersDiv.appendChild(parametersLabel);
                    const parametersInput = document.createElement("input");
                    parametersInput.type = parametersColection[keys[i]];
                    parametersInput.step = "0.0000000001";
                    parametersInput.name = "parameters" + (i + 1);
                    parametersDiv.appendChild(parametersInput);
                    parametersDiv.appendChild(document.createElement("br"));
                }
            }

            if (textClassifier === "SVM") {
                console.log("chegou");
                var parametersColection = {
                    kernel: "text",
                    random_state: "number",
                    C: "number",
                };
                var keys = Object.keys(parametersColection);
                for (var i = 0; i < 3; i++) {
                    console.log(keys[i]);
                    console.log(parametersColection[keys[i]]);
                    const parametersLabel = document.createElement("label");
                    parametersLabel.textContent = keys[i] + ":";
                    parametersDiv.appendChild(parametersLabel);
                    const parametersInput = document.createElement("input");
                    parametersInput.type = parametersColection[keys[i]];
                    parametersInput.step = "0.0000000001";
                    parametersInput.name = "parameters" + (i + 1);
                    parametersDiv.appendChild(parametersInput);
                    parametersDiv.appendChild(document.createElement("br"));
                }
            }

            if (textClassifier === "LOGISTICS REGRESSION") {
                console.log("chegou");
                var parametersColection = {
                    random_state: "number",
                    max_iter: "number",
                    penalty: "text",
                    tol: "number",
                    C: "number",
                    solver: "text",
                };
                var keys = Object.keys(parametersColection);
                for (var i = 0; i < 6; i++) {
                    console.log(keys[i]);
                    console.log(parametersColection[keys[i]]);
                    const parametersLabel = document.createElement("label");
                    parametersLabel.textContent = keys[i] + ":";
                    parametersDiv.appendChild(parametersLabel);
                    const parametersInput = document.createElement("input");
                    parametersInput.type = parametersColection[keys[i]];
                    parametersInput.step = "0.0000000001";
                    parametersInput.name = "parameters" + (i + 1);
                    parametersDiv.appendChild(parametersInput);
                    parametersDiv.appendChild(document.createElement("br"));
                }
            }

            if (textClassifier === "KNN") {
                console.log("chegou");
                var parametersColection = {
                    n_neighbors: "number",
                    metric: "text",
                    p: "number",
                };
                var keys = Object.keys(parametersColection);
                for (var i = 0; i < 3; i++) {
                    console.log(keys[i]);
                    console.log(parametersColection[keys[i]]);
                    const parametersLabel = document.createElement("label");
                    parametersLabel.textContent = keys[i] + ":";
                    parametersDiv.appendChild(parametersLabel);
                    const parametersInput = document.createElement("input");
                    parametersInput.type = parametersColection[keys[i]];
                    parametersInput.step = "0.0000000001";
                    parametersInput.name = "parameters" + (i + 1);
                    parametersDiv.appendChild(parametersInput);
                    parametersDiv.appendChild(document.createElement("br"));
                }
            }

            if (textClassifier === "DECISION TREE") {
                console.log("chegou");
                var parametersColection = {
                    criterion: "text",
                    random_state: "number",
                    max_depth: "text",
                };
                var keys = Object.keys(parametersColection);
                for (var i = 0; i < 3; i++) {
                    console.log(keys[i]);
                    console.log(parametersColection[keys[i]]);
                    const parametersLabel = document.createElement("label");
                    parametersLabel.textContent = keys[i] + ":";
                    parametersDiv.appendChild(parametersLabel);
                    const parametersInput = document.createElement("input");
                    parametersInput.type = parametersColection[keys[i]];
                    parametersInput.step = "0.0000000001";
                    parametersInput.name = "parameters" + (i + 1);
                    parametersDiv.appendChild(parametersInput);
                    parametersDiv.appendChild(document.createElement("br"));
                }
            }

            if (textClassifier === "XGBOOST") {
                console.log("chegou");
                var parametersColection = {
                    max_depth: "number",
                    learning_rate: "number",
                    n_estimators: "number",
                    objective: "text",
                    random_state: "number",
                };
                var keys = Object.keys(parametersColection);
                for (var i = 0; i < 5; i++) {
                    console.log(keys[i]);
                    console.log(parametersColection[keys[i]]);
                    const parametersLabel = document.createElement("label");
                    parametersLabel.textContent = keys[i] + ":";
                    parametersDiv.appendChild(parametersLabel);
                    const parametersInput = document.createElement("input");
                    parametersInput.type = parametersColection[keys[i]];
                    parametersInput.step = "0.0000000001";
                    parametersInput.name = "parameters" + (i + 1);
                    parametersDiv.appendChild(parametersInput);
                    parametersDiv.appendChild(document.createElement("br"));
                }
            }

            if (textClassifier === "LIGHTGBM") {
                console.log("chegou");
                var parametersColection = {
                    num_leaves: "number",
                    objective: "text",
                    max_depth: "number",
                    learning_rate: "number",
                    max_bin: "number",
                    num_boost_round: "number",
                };
                var keys = Object.keys(parametersColection);
                for (var i = 0; i < 6; i++) {
                    console.log(keys[i]);
                    console.log(parametersColection[keys[i]]);
                    const parametersLabel = document.createElement("label");
                    parametersLabel.textContent = keys[i] + ":";
                    parametersDiv.appendChild(parametersLabel);
                    const parametersInput = document.createElement("input");
                    parametersInput.type = parametersColection[keys[i]];
                    parametersInput.step = "0.0000000001";
                    parametersInput.name = "parameters" + (i + 1);
                    parametersDiv.appendChild(parametersInput);
                    parametersDiv.appendChild(document.createElement("br"));
                }
            }

            if (textClassifier === "CATBOOST") {
                console.log("chegou");
                var parametersColection = {
                    task_type: "text",
                    iterations: "number",
                    learning_rate: "number",
                    depth: "number",
                    random_state: "number",
                    eval_metric: "text",
                };
                var keys = Object.keys(parametersColection);
                for (var i = 0; i < 6; i++) {
                    console.log(keys[i]);
                    console.log(parametersColection[keys[i]]);
                    const parametersLabel = document.createElement("label");
                    parametersLabel.textContent = keys[i] + ":";
                    parametersDiv.appendChild(parametersLabel);
                    const parametersInput = document.createElement("input");
                    parametersInput.type = parametersColection[keys[i]];
                    parametersInput.step = "0.0000000001";
                    parametersInput.name = "parameters" + (i + 1);
                    parametersDiv.appendChild(parametersInput);
                    parametersDiv.appendChild(document.createElement("br"));
                }
            }
        });
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
            let text_or_number = lines[1].split(separator);

            console.log("Cabeçalho: ", header);
            const featuresDiv = document.getElementById("features");
            featuresDiv.innerHTML = "";

            for (let i = 0; i < header.length - 1; i++) {
                const featureLabel = document.createElement("label");
                featureLabel.textContent = header[i] + ":";
                featuresDiv.appendChild(featureLabel);
                const featureInput = document.createElement("input");
                if (!isNaN(text_or_number[i])) {
                    featureInput.type = "number";
                    featureInput.step = "0.0000000001";
                } else {
                    featureInput.type = "text";
                }
                featureInput.name = "feature" + (i + 1);
                featuresDiv.appendChild(featureInput);
                featuresDiv.appendChild(document.createElement("br"));
            }
        };

        reader.onerror = function (e) {
            console.error("Erro ao ler o arquivo: ", e);
        };

        reader.readAsText(file);
        });

    document
        .getElementById("prediction-form")
        .addEventListener("submit", function (event) {
            document.getElementById("result").innerText = "";
            event.preventDefault();
            const formData = new FormData(this);
            formData.append("separator", separator);
            console.log(formData);

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
                    console.log(`Previsão: ${data.prediction}`);
                    console.log(`Accuracy test: ${data.accuracy_test}`);
                    console.log(`Accuracy training: ${data.accuracy_training}`);
                    document.getElementById(
                        "result"
                    ).innerText = `Previsão: ${data.prediction}\nAcuracia teste: ${data.accuracy_test}%\nAcuracia treino: ${data.accuracy_training}%`;
                })
                .catch((error) => {
                    console.error("Erro:", error);
                    document.getElementById(
                        "result"
                    ).innerText = `Erro: ${error.message}`;
                });
        });
});
