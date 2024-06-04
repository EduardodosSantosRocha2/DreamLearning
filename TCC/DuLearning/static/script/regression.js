document.addEventListener("DOMContentLoaded", function () {
    var separator = "";
    var csvFileInput = document.getElementById("csv_file");
    var regressionSelect = document.getElementById("regression");
    var parametersDiv = document.getElementById("parameters");
    var featuresDiv = document.getElementById("features");
    var reader;
    var selectedOption = "";

    regressionSelect.addEventListener("change", function () {
        selectedOption = regressionSelect.options[regressionSelect.selectedIndex].text;
        parametersDiv.innerHTML = "";
        if(selectedOption === "SIMPLE LINEAR"){
            hideElement(featuresDiv);
        }else{
            showElement(featuresDiv);
        }
        // Se a opção selecionada for "SIMPLE LINEAR" e houver um arquivo CSV selecionado
        if (selectedOption === "SIMPLE LINEAR" && csvFileInput.files.length > 0) {
            handleCSVFile(csvFileInput.files[0]);
        }
        
       
    });

    csvFileInput.addEventListener("change", function () {
        // Se houver um arquivo CSV selecionado e a opção selecionada for "SIMPLE LINEAR"
        if (regressionSelect.value === "simple_linear_regression" && csvFileInput.files.length > 0) {
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
        var selectHTML = '<label for="csv_headers">Escolha a variavel idepedente:</label>';
        selectHTML += '<select id="csv_headers" name="csv_headers">';

        headers.forEach(function (header) {
            selectHTML += '<option value="' + header + '">' + header + '</option>';
        });

        selectHTML += '</select>';
        parametersDiv.innerHTML = selectHTML;
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
                    ).innerText = `Coeficiente_linear: ${data.Coeficiente_linear}\nCoeficiente_angular: ${data.Coeficiente_angular}\nCoeficiente de determinação do treinamento: ${data.determinationCoefficientTraining}\nCoeficiente de determinação do teste: ${data.determinationCoefficientTest}\nErro absoluto: ${data.abs}\nErro quadrático médio: ${data.MeanSquaredError}`;
                })
                .catch((error) => {
                    console.error("Erro:", error);
                    document.getElementById(
                        "result"
                    ).innerText = `Erro: ${error.message}`;
                });
        });


});

