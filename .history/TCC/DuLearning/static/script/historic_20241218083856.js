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

firebase.auth().onAuthStateChanged(user=>{
    if(user){
        findTransactions(user);
    }
})

function findTransactions(user){
    firebase.firestore()
        .collection('transactions')
        .where('user.uid', '==', user.uid)
        .orderBy('date', 'desc')
        .get()
        .then(snapshot=>{
            transactions = snapshot.docs.map(doc => doc.data());
            createTimeline(transactions);
        })
        .catch(error => {
            console.log(error);
            alert('Erro ao recuperar transaçoes');
        })
}


function formatarParametros(parameters) {
    if (!parameters || Object.keys(parameters).length === 0) {
        return "{}";
    }

    return Object.entries(parameters)
        .map(([key, value]) => `${key} =  ${value}`)
        .join(", ");
}

function createTimeline(transactions) {
    const timeline = document.querySelector('.timeline');
    timeline.innerHTML = ''; 

    transactions.forEach((transaction, index) => {

        const lado = index % 2 === 0 ? 'left-container' : 'right-container';


        const container = document.createElement('div');
        container.className = `containertimeline ${lado}`;

        if(transaction.type == "Classificação"){
            container.innerHTML = `
            <div class="text-box">
                <h2>${transaction.Algorithm || "Sem título"}</h2>
                <h7>${transaction.date}</h7>
                <p>${transaction.type}</p>
                <h7>No modelo de ${transaction.type || "nenhum modelo de classificação"} aplicado à base de dados "${transaction.NameDatabase || "base de dados desconhecida"}", o algoritmo ${transaction.Algorithm || "Sem título"}, configurado com os parâmetros ${formatarParametros(transaction.Parameters) || "Sem parametros"}, obteve ${transaction.TrainingAccuracy || "Sem acuracia de treino"}% de acurácia no treinamento, ${transaction.TestAccuracy || "Sem acuracia de teste"}% no teste e ${transaction.CrossValidation || "Sem validação cruzada"}% na validação cruzada.</h7>
                <div class="icon-buttons">
                    <button type="button" onclick="downloadFile()" style="border: none;" title="Baixar arquivo CSV">
                        <i class='bx bx-cloud-download'></i>
                    </button>
                    <button type="button" onclick="copyToClipboard()" style="border: none;" title="Copiar o código-fonte KNN">
                        <i class='bx bx-copy-alt'></i>
                    </button>
                </div>
                <span class="${lado}-arrow"></span>
            </div>
        `;
        }else if(transaction.type == "Regressão"){
            container.innerHTML = `
            <div class="text-box">
                <h2>${transaction.Algorithm || "Sem título"}</h2>
                <h6>${transaction.date}</h6>
                <h>${transaction.type}</h>
                <h5>No modelo de ${transaction.type || "nenhum modelo de regressão"} aplicado à base de dados "${transaction.NameDatabase || "base de dados desconhecida"}", o algoritmo ${transaction.Algorithm || "sem título"}, configurado com os parâmetros ${formatarParametros(transaction.Parameters) || "{}"}, obteve um coeficiente de determinação do treinamento ${transaction.CoefficientTraining || "sem coeficiente de treino"}% e de teste ${transaction.CoefficientTest || "sem coeficiente de teste"}%, um erro médio absoluto de  ${transaction.abs || "sem erro médio absoluto"} e raiz erro quadrático médio de ${transaction.MeanSquaredError || "sem erro quadrático médio"}.</h5>
                <div class="icon-buttons">
                    <button type="button" onclick="downloadFile()" style="border: none;" title="Baixar arquivo CSV">
                        <i class='bx bx-cloud-download'></i>
                    </button>
                    <button type="button" onclick="copyToClipboard()" style="border: none;" title="Copiar o código-fonte KNN">
                        <i class='bx bx-copy-alt'></i>
                    </button>
                </div>
                <span class="${lado}-arrow"></span>
            </div>
        `;
        }
        

        timeline.appendChild(container);
        
    });
}



