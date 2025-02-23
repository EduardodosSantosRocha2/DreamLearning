document.addEventListener('DOMContentLoaded', function() {
    let left_btn = document.getElementsByClassName('bi-chevron-left')[0];
    let right_btn = document.getElementsByClassName('bi-chevron-right')[0];
    let cards = document.getElementsByClassName('cards')[0];

    left_btn.addEventListener('click', () => {
        cards.scrollLeft -= 140;
    });

    right_btn.addEventListener('click', () => {
        cards.scrollLeft += 140;
    });

    let json_url = "/static/json/movie.json";
    console.log(json_url);


    fetch(json_url).then(response => response.json())
        .then((data) => {
            data.forEach((ele) => {
                let { name, type, date, sposter, bposter, genre, url } = ele;
                let card = document.createElement('a');
                card.classList.add('card1');
                card.href = url;
                card.innerHTML = `
                    <img src="${sposter}" alt="${name}" class="poster">
                    <div class="rest_card">
                        <img src="${bposter}" alt="">
                        <div class="cont">
                            <h4>${name}</h4>
                            <div class="sub">
                                <p>${genre}, ${date}</p>
                                <h3><span>IA</span><i class="bi bi-star-fill"></i>${type}</h3>
                            </div>
                        </div>
                    </div>`;
                cards.appendChild(card);
            });

            // Filtragem de tipo   
            function filtragem(type, text){
                console.log(text);
                let typeAlg = document.getElementById(type);
                typeAlg.addEventListener('click', () => {
                cards.innerHTML = '';
                
                if(type === "classificao" || type === "regressao"){
                    var typeAlg_array = data.filter(ele => ele.type === text || ele.conttypes === 2);
                }else{
                    var typeAlg_array = data.filter(ele => ele.type === text);
                    console.log(typeAlg_array);
                }
                
                typeAlg_array.forEach((ele) => {
                    
                    let { name, type, date, sposter, bposter, genre, url } = ele;
                    console.log(type);
                    let card = document.createElement('a');
                    card.classList.add('card1');
                    card.href = url;
                    card.innerHTML = `
                        <img src="${sposter}" alt="${name}" class="poster">
                        <div class="rest_card">
                            <img src="${bposter}" alt="">
                            <div class="cont">
                                <h4>${name}</h4>
                                <div class="sub">
                                    <p>${genre}, ${date}</p>
                                    <h3><span>IA</span><i class="bi bi-star-fill"></i>${type}</h3>
                                </div>
                            </div>
                        </div>`;
                    cards.appendChild(card);
                });
            });
            }

            filtragem("classificao", "Classificação");
            filtragem("regressao", "Regressão");
            filtragem("associacao", "Associação");
            
        });
});
