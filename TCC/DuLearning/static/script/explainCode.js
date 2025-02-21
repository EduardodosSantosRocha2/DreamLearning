const buttonExplainCode = document.getElementById("explainCode");


const sendChatMessage = async () => {
    const code = document.getElementById("windowcode").textContent;
    const modalBody = document.getElementsByClassName("modal-body")[0];
    modalBody.innerHTML ="";
    
    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: `Explique resumidamente analisando linha a linha do código:  ${code}` }),
        });

        const data = await response.json();
        console.log(data.bot_reply)
        const div = document.createElement('div');
        div.className = "message bot"
        div.innerHTML = data.bot_reply;
        modalBody.appendChild(div);


    } catch (error) {
        modalBody.innerHTML = "Erro ao conectar com o servidor.";
    }
};


function explainCode() {
    let userMessageInput = document.getElementById("windowcode")?.value || "";
    console.log(userMessageInput);
    sendChatMessage();
}



buttonExplainCode.addEventListener("click", () => {
    explainCode();
});
