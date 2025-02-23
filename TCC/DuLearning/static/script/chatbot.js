const chatbotButton = document.getElementById("chatbot-button");
const chatbotContainer = document.getElementById("chatbot-container");
const closeChatbotButton = document.getElementById("close-chatbot");
const sendButton = document.getElementById("send-button");
const userMessageInput = document.getElementById("user-message");
const chatContent = document.getElementById("chatbot-content");



function pegarSelecionados() {
    let selecionados = [];
    document.querySelectorAll('input[type="checkbox"]:checked').forEach(checkbox => {
        selecionados.push(checkbox.nextElementSibling.textContent);
    });
    return(selecionados.join("\n"))
}

function clearcheckbox() {
    document.querySelectorAll('input[type="checkbox"]:checked').forEach(checkbox => {
        checkbox.checked = false;
    });
}

const addMessage = (message, sender) => {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender);
    messageDiv.innerHTML = message;
    chatContent.appendChild(messageDiv);
    var type = "";
    if (sender === "user"){
        type = 'end';
    }else{
        type = 'start';
    }
    
    scrollToLastMessage(sender, type);
};



// Função para enviar mensagem ao servidor
const sendMessage = async () => {
    var userMessage =""
    checkboxChecked = pegarSelecionados();
    if(checkboxChecked!=""){
        userMessage = checkboxChecked
        clearcheckbox()
    }else{
        userMessage = userMessageInput.value.trim();
        addMessage(userMessage, "user");
        userMessageInput.value = "";
    }
    
    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: `Em no maximo 600 caracteres explique sobre: ${userMessage} `}),
        });
        const data = await response.json();
        addMessage(data.bot_reply, "bot");
    } catch (error) {
        addMessage("Erro ao conectar com o servidor.", "bot");
    }
};

closeChatbotButton.addEventListener("click", () => {
    chatbotContainer.style.display = "none"; // Esconde o chatbot
    chatbotButton.style.display = "flex";   // Mostra o botão para reabrir
});

chatbotButton.addEventListener("click", () => {
    chatbotContainer.style.display = "flex"; // Mostra o chatbot
    chatbotButton.style.display = "none";    // Esconde o botão de reabertura
    

});

sendButton.addEventListener("click", sendMessage);

userMessageInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
});

window.onload = () => {
    chatbotContainer.style.display = "flex";
    chatbotButton.style.display = "none";
};

function scrollToLastMessage(sender, type) {
    const lastMessage = document.querySelector(`.message.${sender}:last-child`);
    if (lastMessage) {
        lastMessage.scrollIntoView({ behavior: 'smooth', block: type });
    }
}


