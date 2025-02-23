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

