document.addEventListener("DOMContentLoaded", function () {
    var separator = ""; // Define separator como variável global

    document.getElementById("csv_file").addEventListener("change", function (event) {
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
            console.log("Esse é o separator" + separator)
        };

        reader.readAsText(file);
    });

    document.getElementById("prediction-form").addEventListener("submit", function (event) {
        document.getElementById("result").innerText = "";
        event.preventDefault();
        const formData = new FormData(this);
        formData.append("separator", separator); 
        console.log(formData);

        fetch("/typedatatest", {
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
                var continuesNormality = "";
                console.log(`linearCorrelation: ${data.linearCorrelation}`);
                console.log(`continuesNormality: ${data.continuesNormality}`);
                if (!data.continuesNormality) {
                    continuesNormality = "A base de dados não possui distribuição normal"
                } else {
                    continuesNormality = "A base de dados possui distribuição normal"
                }
                document.getElementById("result").innerText = `${continuesNormality}`;

                var json_correlacoes = data.linearCorrelation;

                // Converter o JSON para um objeto JavaScript
                var correlacoes = JSON.parse(json_correlacoes);

                // Obter os cabeçalhos das colunas
                var colunas = Object.keys(correlacoes);

                // Obter o cabeçalho da tabela
                var headerRow = document.getElementById('headerRow');
                headerRow.innerText = "";

                // Obter o corpo da tabela
                var tableBody = document.getElementById('tableBody');
                tableBody.innerText = "";

                // Criar os cabeçalhos das colunas dinamicamente
                var emptyHeaderCell = document.createElement('th'); // Célula vazia para o espaço no canto superior esquerdo
                headerRow.appendChild(emptyHeaderCell);
                colunas.forEach(function (coluna) {
                    var th = document.createElement('th');
                    th.textContent = coluna;
                    headerRow.appendChild(th);
                });

                
                

                // Iterar sobre as colunas
                colunas.forEach(function (coluna) {
                    // Criar uma nova linha para cada coluna
                    var tr = document.createElement('tr');

                    // Adicionar o cabeçalho da linha
                    var th = document.createElement('th');
                    th.textContent = coluna;
                    tr.appendChild(th);

                    // Obter os valores para a coluna atual
                    var valores = correlacoes[coluna];

                    // Iterar sobre as colunas novamente para obter os valores
                    colunas.forEach(function (outraColuna) {
                        // Adicionar o valor à linha
                        var td = document.createElement('td');
                        td.textContent = valores[outraColuna];
                        tr.appendChild(td);
                    });

                    // Adicionar a linha ao corpo da tabela
                    tableBody.appendChild(tr);
                });

            })
            .catch((error) => {
                console.error("Erro:", error);
                document.getElementById("result").innerText = `Erro: ${error.message}`;
            });
    });
});
