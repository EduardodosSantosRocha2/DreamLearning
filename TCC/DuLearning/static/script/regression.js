document.addEventListener("DOMContentLoaded", function () {
    var csvFileInput = document.getElementById("csv_file");
    var regressionSelect = document.getElementById("regression");
    var parametersDiv = document.getElementById("parameters");
    var reader;

    regressionSelect.addEventListener("change", function () {
        var selectedOption = regressionSelect.options[regressionSelect.selectedIndex].text;

        // Limpa o parâmetros sempre que o tipo de regressão mudar
        parametersDiv.innerHTML = "";

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

    document
        .getElementById("prediction-form")
        .addEventListener("submit", function (event) {
            document.getElementById("result").innerText = "";
            event.preventDefault();
            const formData = new FormData(this);
            // formData.append("separator", separator);
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
                    console.log(`selected_Independent_Variable: ${data.select_Independent_Variable}`);
                    document.getElementById(
                        "result"
                    ).innerText = `Variavel idepedente: ${data.select_Independent_Variable}`;
                })
                .catch((error) => {
                    console.error("Erro:", error);
                    document.getElementById(
                        "result"
                    ).innerText = `Erro: ${error.message}`;
                });
        });






});

