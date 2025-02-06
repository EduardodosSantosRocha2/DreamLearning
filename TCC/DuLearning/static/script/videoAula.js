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

document.addEventListener('DOMContentLoaded', function() {
    let left_btn = document.getElementsByClassName('bi-chevron-left')[0];
    let right_btn = document.getElementsByClassName('bi-chevron-right')[0];
    let cards = document.getElementsByClassName('cards')[0];
    let search = document.getElementsByClassName('search')[0];
    let search_input = document.getElementById('search_input');

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

            // Carregar dados de busca
            data.forEach(ele => {
                let { name, type, date, sposter, genre, url } = ele;
                let card = document.createElement('a');
                card.classList.add('card1');
                card.href = url;
                card.innerHTML = `
                    <img src="${sposter}" alt="">
                    <div class="cont">
                        <h3>${name}</h3>
                        <p>${genre}, ${date}, <span>IA</span><i class="bi bi-star-fill"></i> ${type}</p>
                    </div>`;
                search.appendChild(card);
            });

            // Filtro de busca
            search_input.addEventListener('keyup', () => {
                let filter = search_input.value.toUpperCase();
                let a = search.getElementsByTagName('a');
                for (let i = 0; i < a.length; i++) {
                    let b = a[i].getElementsByClassName('cont')[0];
                    let textValue = b.textContent || b.innerText;
                    if (textValue.toUpperCase().indexOf(filter) > -1) {
                        a[i].style.display = "flex";
                        search.style.visibility = "visible";
                        search.style.opacity = 1;
                    } else {
                        a[i].style.display = "none";
                    }
                    if (search_input.value == 0) {
                        search.style.visibility = "hidden";
                        search.style.opacity = 0;
                    }
                }
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
