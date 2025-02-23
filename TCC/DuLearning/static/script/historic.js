
var u  = "";
var transactions = ""


firebase.auth().onAuthStateChanged(user=>{
    if(user){
        u = user;
        findTransactions(user);
    }
})

function findTransactions(user) {
    firebase.firestore()
        .collection('transactions')
        .where('user.uid', '==', user.uid)
        .orderBy('date', 'desc')
        .get()
        .then(snapshot => {
            transactions = snapshot.docs.map(doc => ({
                id: doc.id, 
                ...doc.data() 
            }));
            createTimeline(transactions);
        })
        .catch(error => {
            console.log(error);
            alert('Erro ao recuperar transações');
        });
}


function formatarParametros(parameters) {
    if (!parameters || Object.keys(parameters).length === 0) {
        return "{}";
    }

    return Object.entries(parameters)
        .map(([key, value]) => `<span class="key">${key}</span> =  <span class="value">${value}</span>`)
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
                <h7>No modelo de ${transaction.type || "nenhum modelo de classificação"} aplicado à base de dados "${transaction.NameDatabase || "base de dados desconhecida"}", o algoritmo ${transaction.Algorithm || "Sem título"}, configurado com os parâmetros ${formatarParametros(transaction.Parameters) || "Sem parametros"}, obteve <span class="percents">${transaction.TrainingAccuracy || "Sem acuracia de treino"}%</span> de acurácia no treinamento, <span class="percents">${transaction.TestAccuracy || "Sem acuracia de teste"}%</span> no teste e <span class="percents">${transaction.CrossValidation || "Sem validação cruzada"}%</span> na validação cruzada.</h7>
                <div class="icon-buttons">
                    <button type="button" value = ${transaction.id} onclick="downloadFile(this)" style="border: none;" title="Baixar arquivo CSV">
                        <i class='bx bx-cloud-download'></i>
                    </button>
                    <button type="button" value = ${transaction.id} onclick="copyToClipboard(this)" style="border: none;" title="Copiar o código-fonte">
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
                <h7>${transaction.date}</h7>
                <p>${transaction.type}</p>
                <h7>No modelo de ${transaction.type || "nenhum modelo de regressão"} aplicado à base de dados "${transaction.NameDatabase || "base de dados desconhecida"}", o algoritmo ${transaction.Algorithm || "sem título"}, configurado com os parâmetros ${formatarParametros(transaction.Parameters) || "{}"}, obteve um coeficiente de determinação do treinamento <span class="percents">${transaction.CoefficientTraining || "sem coeficiente de treino"}%</span> e de teste <span class="percents">${transaction.CoefficientTest || "sem coeficiente de teste"}%</span>, um erro médio absoluto de  <span class="percents">${transaction.abs || "sem erro médio absoluto"}</span>, raiz erro quadrático médio de <span class="percents">${transaction.MeanSquaredError || "sem erro quadrático médio"}</span> e validação cruzada <span class="percents">${transaction.CrossValidation || "sem erro quadrático médio"}%</span>.</h7>
                <div class="icon-buttons">
                    <button type="button" value = ${transaction.id} onclick="downloadFile(this)" style="border: none;" title="Baixar arquivo CSV">
                        <i class='bx bx-cloud-download'></i>
                    </button>
                    <button type="button" value = ${transaction.id} onclick="copyToClipboard(this)" style="border: none;" title="Copiar o código-fonte">
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


async function downloadFile(button){
    let base64Data =""; 
    let namebase = ""

    transactions.forEach((transaction) => {
        if(transaction.id === button.value){
            base64Data = transaction.csv;
            namebase = transaction.NameDatabase;
            return;
        }
    });
    
    const dataUrl = `data:text/csv;base64,${base64Data}`;
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = namebase;  
    link.click();  
    link.remove();
}


function copyToClipboard(event){
    let code = ""
    transactions.forEach((transaction) => {
        if(transaction.id === event.value){
            code = atob(transaction.code);
            return;
        }
    });
    
    
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






