document.onreadystatechange = function() {
    if(document.readyState != "complete") {
        let items = document.getElementsByClassName("history-item-score");
        for(let i = 0; i < items.length; i++) {
            let score = Number.parseInt(items[i].children[0].innerHTML);
            if(score < 25) {
                items[i].children[0].style.color = 'darkgreen';
            } else if(score < 50) {
                items[i].children[0].style.color = 'rgb(167, 110, 0)';
            } else if(score < 75) {
                items[i].children[0].style.color = '#d35400';
            } else {
                items[i].children[0].style.color = '#c0392b';
            }
        }
    }
}