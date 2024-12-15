function logout(){
    firebase.auth().signOut().then(()=>{
        window.location.href = "/"
    }).catch(error=>{
        alert('Erro a fazer logout!')
    })
}