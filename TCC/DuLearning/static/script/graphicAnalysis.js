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
        scene.appendChild(star)
        i++;
    }
}
stars();


let csvFile = null;

document.getElementById('csvFileInput').addEventListener('change', function(event) {
    csvFile = event.target.files[0];
    if (csvFile) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const text = e.target.result;
            processCSV(text);
        };
        reader.readAsText(csvFile);
    }
});

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
        graphSelect.className = "nes-select"
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

document.getElementById('showSelections').addEventListener('click', function() {
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
    
    const length = Object.keys(selections).length
    console.log("Tamanho: " +length);

    if (csvFile) {
        const formData = new FormData();
        formData.append('csvFile', csvFile);
        formData.append('selections', JSON.stringify(selections));

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
            for(const key in data){
                if(data.hasOwnProperty(key)){
                    console.log(`${key}: `+ data[key])
                    var graphData = JSON.parse(data[key]);
                    console.log(graphData)
                    var innerDiv = document.createElement('div');  
                    var nameDiv = 'inner-div'+i
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
