function toggleDetails() {
    let details = document.getElementById('details-content');
    let toggle = document.getElementById('details-flip')
    if(details.style.display == 'none' || details.style.display == '') {
      details.style.visibility = 'visible'
      details.style.display = 'block';
      toggle.classList.remove('flip');
    } else {
      details.style.visibility = 'hidden'
      details.style.display = 'none';
      toggle.classList.add('flip');
    }
  }