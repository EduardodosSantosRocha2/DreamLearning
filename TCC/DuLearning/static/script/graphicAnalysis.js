let spinner = document.getElementById('spinner');
spinner.style.display = 'none';

let csvFile = null;
let csvContent = null;
let boolean  = true;
let separator = "";




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
        pushNotify('success', 'Sucesso: ', "Código copiado para a área de transferência.");
    }, () => {
        pushNotify('error', 'Falha: ', "Impossivel copiar para a área de transferência.");
    });
}

function pushNotify(status,title,text){

    Notiflix.Notify.init({
        timeout: 2000,  
        fontSize: '16px', 
        useIcon: false, 
        messageMaxLength: 200,  
        position: 'right-top', 
        success: {
            background: '#DFF6DD',
            textColor: '#000000'  
        },
        failure: {
            background: '#F8D7DA', 
            textColor: '#000000' 
        }
    });
    
    if(status === "success"){
        Notiflix.Notify.success(title+text);
    }else{
        Notiflix.Notify.failure(title+text);
    }
}


function clearOptionsContainer() {
    var div = document.getElementById("optionsContainer");
    if (div) {
        div.innerHTML = "";
    }
}

function processCSV(csvText) {
    const lines = csvText.split('\n');
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
    console.log("Meu separator"+ separator)
    const headers = lines[0].split(separator);
    // headers.pop();  // Remove the last column

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
    document.getElementById("result").innerText = "";
    spinner.style.display = 'block';
    var graphDataDiv = document.getElementById('graphData');
    graphDataDiv.innerHTML = '';
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
        formData.append('separator', separator);

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
                spinner.style.display = 'none';
                graphDataDiv.innerHTML = '';
                var code = data.code;
                var data = data.data
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
                document.getElementById("windowcode").innerHTML = code;
            })
            .catch((error) => {
                console.error("Erro:", error);
                document.getElementById("result").innerText = `Erro: ${error.message}`;
            });
    }
});


function processCSVgraphicAnalysis(csvText) {
    const lines = csvText.split('\n');
    // Verifica se a primeira linha contém uma vírgulaa
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
    const headers = lines[0].split(separator);
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
