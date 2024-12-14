let spinner = document.getElementById('spinner');
spinner.style.display = 'none';
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
    var csvFileInput = document.getElementById("csv_file");
    var regressionSelect = document.getElementById("regression");
    var parametersDiv = document.getElementById("parameters");
    var featuresDiv = document.getElementById("features");
    var resultDiv = document.getElementById("result");
    var posicaoDiv = document.getElementById("posicao");
    var reader;
    var selectedOption = "";
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
                parametersInput.classList.add("nes-input");
                parametersInput.className  = "nes-input";
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
                parametersInput.className  = "nes-input";
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
                parametersInput.className  = "nes-input";
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
                parametersInput.className  = "nes-input";
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
                parametersInput.className  = "nes-input";
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
                parametersInput.className  = "nes-input";
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
                    posicaoDiv.textContent = "feature" + (i + 1);
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
            //     let text_or_number = lines[1].split(separator);

            //     console.log("Cabeçalho: ", header);

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

              reader.readAsText(file);
        });

    document
        .getElementById("prediction-form")
        .addEventListener("submit", function (event) {
            document.getElementById(
                "result"
            ).innerText = "";
            
            spinner.style.display = 'block';
            document.getElementById("result").innerText = "";

            event.preventDefault();
            const formData = new FormData(this);
            formData.append("separator", separator);
            formData.append("posicao", document.getElementById("posicao").textContent)
            formData.append("deployBoolean",deployBoolean)
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
                    spinner.style.display = 'none';
                    console.log(`Coeficiente_linear: ${data.Coeficiente_linear}`);
                    let resultText = `Coeficiente de determinação do treinamento: ${data.determinationCoefficientTraining}%<br>`;
                    resultText += `Coeficiente de determinação do teste: ${data.determinationCoefficientTest}%<br>`;
                    resultText += `Erro médio absoluto: ${data.abs}<br>`;
                    resultText += `Raiz erro quadrático médio: ${data.MeanSquaredError}`;
                    resultText1 = "";
                    
                    // Verifica se data.prediction não é vazio antes de adicionar ao texto resultante
                    if (data.prediction !== undefined && data.prediction !== null) {
                        resultText1 = `Previsão: ${data.prediction}<br>`;
                    }
                
                    document.getElementById("result").innerHTML = `<div class="preformatted-text">${resultText}</div>`;
                    document.getElementById("result1").innerHTML = `<div class="preformatted-text">${resultText1}</div>`;
                    
                    document.getElementById("windowcode").innerHTML = data.code;
                })                                           
                .catch((error) => {
                    console.error("Erro:", error);
                    document.getElementById(
                        "result"
                    ).innerText = `Erro: ${error.message}`;
                });
        });


});

