let csvFile = null;
let csvContent = null;
let boolean  = true;

function stars() {
    let count = 500;
    let scene = document.querySelector(".scene");
    let i = 0;
    while (i < count) {
        let star = document.createElement("i");
        let x = Math.floor(Math.random() * window.innerWidth);
        let y = Math.floor(Math.random() * window.innerHeight);
        let duration = Math.random() * 10;
        let size = Math.random() * 2;
        star.style.left = x + 'px';
        star.style.top = y + 'px';
        star.style.width = 1 + size + 'px';
        star.style.height = 1 + size + 'px';
        star.style.animationDuration = 5 + duration + 's';
        star.style.animationDelay = duration + 's';
        scene.appendChild(star);
        i++;
    }
}
stars();

function clearOptionsContainer() {
    var div = document.getElementById("optionsContainer");
    if (div) {
        div.innerHTML = "";
    }
}

function processCSV(csvText) {
    const lines = csvText.split('\n');
    const headers = lines[0].split(',');
    headers.pop();  // Remove the last column

    const container = document.getElementById('optionsContainer');
    container.innerHTML = '';

    headers.forEach(header => {
        const div = document.createElement('div');
        div.className = "nes-select";

        const label = document.createElement('label');
        label.textContent = header;
        div.appendChild(label);

        const graphSelect = document.createElement('select');
        graphSelect.className = "nes-select";
        const graphOptions = ['Gráficos de Linha', 'Gráficos de Dispersão', 'Gráficos de Barras', "Histogramas", "Gráficos de Pizza", "BoxPlot"];
        graphOptions.forEach(option => {
            const opt = document.createElement('option');
            opt.value = option.toLowerCase();
            opt.textContent = option;
            graphSelect.appendChild(opt);
        });
        div.appendChild(graphSelect);

        container.appendChild(div);
    });
}

document.getElementById('csvFileInput').addEventListener('change', function (event) {
    csvFile = event.target.files[0];
    if (csvFile && boolean == true) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const text = e.target.result;
            csvContent = text; // Armazena o conteúdo do CSV
            processCSV(text);
        };
        reader.readAsText(csvFile);
    }
    if(csvFile && boolean == false){
        const reader = new FileReader();
        reader.onload = function (e) {
            const text = e.target.result;
            csvContent = text; // Armazena o conteúdo do CSV
            processCSVgraphicAnalysis(text);
        };
        reader.readAsText(csvFile);
    }

});

document.getElementById("showSelections").addEventListener('click', function () {
    var graphDataDiv = document.getElementById('graphData');
    const container = document.getElementById('optionsContainer');
    const selections = {};

    const divs = container.getElementsByTagName('div');
    Array.from(divs).forEach(div => {
        const label = div.getElementsByTagName('label')[0].textContent;
        const select = div.getElementsByTagName('select')[0];
        const selectedValue = select.value;
        selections[label] = selectedValue;
    });

    console.log("selections:\n")
    console.log(selections)

    const length = Object.keys(selections).length;
    console.log("Tamanho: " + length);

    if (csvFile) {
        const formData = new FormData();
        formData.append('csvFile', csvFile);
        formData.append('selections', JSON.stringify(selections));
        formData.append('typeGraphic', boolean);

        fetch('/graphicAnalysisPost', {
            method: 'POST',
            body: formData
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error("Erro ao fazer a solicitação.");
                }
                return response.json();
            })
            .then((data) => {
                graphDataDiv.innerHTML = '';
                let i = 1;
                for (const key in data) {
                    if (data.hasOwnProperty(key)) {
                        console.log(`${key}: ` + data[key]);
                        var graphData = JSON.parse(data[key]);
                        console.log(graphData);
                        var innerDiv = document.createElement('div');
                        var nameDiv = 'inner-div' + i;
                        innerDiv.className = nameDiv;
                        graphDataDiv.appendChild(innerDiv);

                        // Plotar os gráficos
                        Plotly.react(innerDiv, graphData.data, graphData.layout);
                        i++;
                    }
                }
            })
            .catch((error) => {
                console.error("Erro:", error);
                document.getElementById("result").innerText = `Erro: ${error.message}`;
            });
    }
});


function processCSVgraphicAnalysis(csvText) {
    const lines = csvText.split('\n');
    const headers = lines[0].split(',');
    headers.pop(); 

    const container = document.getElementById('optionsContainer');
    container.innerHTML = '';

    const graphOptions = headers;
    const chartTypes = [
        'Gráficos de Linha',
        'Gráficos de Dispersão',
        'Gráficos de Barras',
        'Histogramas',
        'Gráficos de Pizza',
        'BoxPlot'
    ];

    
    function createDropdown(labelText, options) {
        const div = document.createElement('div');
        div.className = "nes-select";
        
        const label = document.createElement('label');
        label.textContent = labelText;
        div.appendChild(label);

        const select = document.createElement('select');
        select.className = "nes-select";
        options.forEach(option => {
            const opt = document.createElement('option');
            opt.value = option;
            opt.textContent = option;
            select.appendChild(opt);
        });
        div.appendChild(select);

        return div;
    }

    
    container.appendChild(createDropdown("Escolha o Tipo de Gráfico", chartTypes));

    
    container.appendChild(createDropdown("Escolha a Variável 1", graphOptions));

    
    container.appendChild(createDropdown("Escolha a Variável 2", graphOptions));
}





document.getElementById("graphic").addEventListener("change", function (event) {
    var select = document.querySelector("#graphic");
    var option = select.children[select.selectedIndex];
    var textClassifier = option.textContent;
    console.log(textClassifier);

    if (textClassifier === "Análise Gráfica Completa") {
        boolean = true;
        clearOptionsContainer(); 

        if (csvContent) {
            processCSV(csvContent); 
        }
    } else if (textClassifier === "Análise Gráfica Duas Variaveis") {
        boolean = false;
        clearOptionsContainer();
        if (csvContent) {
            processCSVgraphicAnalysis(csvContent)
        }
         
    }
});
