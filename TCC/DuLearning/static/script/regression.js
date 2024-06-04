document.addEventListener("DOMContentLoaded", function () {
    var separator = "";
    var csvFileInput = document.getElementById("csv_file");
    var regressionSelect = document.getElementById("regression");
    var parametersDiv = document.getElementById("parameters");
    var featuresDiv = document.getElementById("features");
    var resultDiv = document.getElementById("result");
    var posicaoDiv = document.getElementById("posicao");
    var reader;
    var selectedOption = "";

    regressionSelect.addEventListener("change", function () {
        resultDiv.innerHTML = "";
        selectedOption = regressionSelect.options[regressionSelect.selectedIndex].text;
        parametersDiv.innerHTML = "";


        // if (selectedOption === "SIMPLE LINEAR" || selectedOption === "POLYNOMIAL") {
        //     hideElement(featuresDiv);
        // } else {
        //     showElement(featuresDiv);
        // }
        // Se a opção selecionada for "SIMPLE LINEAR" e houver um arquivo CSV selecionado
        if (selectedOption === "SIMPLE LINEAR" || selectedOption === "POLYNOMIAL" && csvFileInput.files.length > 0) {
            handleCSVFile(csvFileInput.files[0]);
        }
        else if (selectedOption === "SUPPORT VECTORS(SVR)") {
            console.log("chegou");
            var parametersColection = {
                kernel: "text"
            };
            var keys = Object.keys(parametersColection);
            for (var i = 0; i < 1; i++) {
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
        else if (selectedOption === "DECISION TREE") {
            console.log("chegou");
            var parametersColection = {
                max_depth: "number",
                random_state: "number"
            };
            var keys = Object.keys(parametersColection);
            for (var i = 0; i < 2; i++) {
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
        else if (selectedOption === "RANDOM FOREST") {
            console.log("chegou");
            var parametersColection = {
                n_estimators: "number",
                criterion: "text",
                max_depth: "number",
                random_state: "number"
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
        else if (selectedOption === "XGBOOST") {
            console.log("chegou xg");
            var parametersColection = {
                n_estimators: "number",
                max_depth: "number",
                learning_rate: "number",
                objective: "text",
                random_state: "number"
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
        else if (selectedOption === "LIGHT GBM") {
            console.log("chegou");
            var parametersColection = {
                num_leaves: "number",
                max_depth: "number",
                learning_rate: "number",
                n_estimators: "number",
                random_state: "number"
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

        else if (selectedOption === "CATBOOST") {
            console.log("chegou");
            var parametersColection = {
                iterations: "number",
                learning_rate: "number",
                depth: "number",
                random_state: "number",

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
    });

    csvFileInput.addEventListener("change", function () {
        // Se houver um arquivo CSV selecionado e a opção selecionada for "SIMPLE LINEAR"
        if (regressionSelect.value === "simple_linear_regression" || regressionSelect.value === "polynomial_regression" && csvFileInput.files.length > 0) {
            handleCSVFile(csvFileInput.files[0]);
        }
    });

    function handleCSVFile(file) {
        reader = new FileReader();
        reader.onload = function (e) {
            var contents = e.target.result;
            var lines = contents.split("\n");
            if (lines.length > 0) {
                // Verifica o separador
                var separator = ",";
                if (lines[0].includes(",")) {
                    separator = ",";
                } else if (lines[0].includes(";")) {
                    separator = ";";
                } else {
                    console.error("Separador não reconhecido.");
                    return;
                }

                var headers = lines[0].split(separator);
                populateHeaders(headers);
            }
        };

        reader.readAsText(file);
    }

    function populateHeaders(headers) {
        var selectHTML = '<label for="csv_headers">Escolha a variável independente:</label>';
        selectHTML += '<select id="csv_headers" name="csv_headers">';
    
        headers.forEach(function (header) {
            selectHTML += '<option value="' + header + '">' + header + '</option>';
        });
    
        selectHTML += '</select>';
        parametersDiv.innerHTML = selectHTML;
    
        const Ivariable = document.getElementById("csv_headers");
        var options = Ivariable.options;
    
        // Adiciona o texto inicial na div posicaoDiv
        posicaoDiv.textContent = "feature1";
    
        // Adiciona o evento para atualizar a posição selecionada
        Ivariable.addEventListener("change", function () {
            for (var i = 0; i < options.length; i++) {
                if (options[i].selected) {
                    posicaoDiv.textContent = "feature" + (i+1);
                    break;
                }
            }
        });
    }
    
    
    

    function hideElement(element) {
        element.style.display = "none";
    }

    function showElement(element) {
        element.style.display = "block";
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
                let text_or_number = lines[1].split(separator);

                console.log("Cabeçalho: ", header);

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
            formData.append("posicao", document.getElementById("posicao").textContent)
            console.log(formData);
            fetch("/regressionPost", {
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
                    console.log(`Coeficiente_linear: ${data.Coeficiente_linear}`);
                    document.getElementById(
                        "result"
                    ).innerText = `prediction:${data.prediction}\nCoeficiente de determinação do treinamento: ${data.determinationCoefficientTraining}\nCoeficiente de determinação do teste: ${data.determinationCoefficientTest}\nErro absoluto: ${data.abs}\nErro quadrático médio: ${data.MeanSquaredError}`;
                })
                .catch((error) => {
                    console.error("Erro:", error);
                    document.getElementById(
                        "result"
                    ).innerText = `Erro: ${error.message}`;
                });
        });


});

