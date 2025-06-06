firebase.auth().onAuthStateChanged(user=>{
    if (user){
        window.location.href ="/Home"
    }    
})


const container = document.querySelector('.container')
const registerBtn = document.querySelector('.register-btn')
const loginBtn = document.querySelector('.login-btn')

registerBtn.addEventListener('click', ()=>{
    container.classList.add('active');

})


loginBtn.addEventListener('click', ()=>{
    container.classList.remove('active');
})

const form = {
    email: () => document.getElementById("email"),
    password: () => document.getElementById("password"),
    emailnew: ()=> document.getElementById("emailnew"),
    passwordnew: () => document.getElementById("passwordnew"),
} 


function login(){
    firebase.auth().signInWithEmailAndPassword(form.email().value, form.password().value).then(response => {
        window.location.href ="/Home"
    }).catch(error => {
        console.log(error)
        alert(getErrorMessage(error))
    });
}

function getErrorMessage(error){
    if(error.code =="auth/invalid-credential"){
        return "Credenciais invalidas!"
    }
    if(error.code =="auth/email-already-in-use"){
        return "Email já cadastrado!"
    }
    
    
}

function recoverPassword(){
    firebase.auth().sendPasswordResetEmail(form.email().value).then(()=>{
        alert('Email de recuperação para '+form.email().value+' enviado!');
    }).catch(error =>{
        console.log(error);
        alert('Erro para recuperar a senha. Tente Novamente.');
    });
}

function register(){
    firebase.auth().createUserWithEmailAndPassword(form.emailnew().value, form.passwordnew().value).then(()=>{
         window.location.href ="/Home";
    }).catch(error =>{
        console.log(error);
        alert(getErrorMessage(error));
    })
}




