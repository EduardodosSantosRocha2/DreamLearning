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