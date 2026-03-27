document.onreadystatechange = function() {
  if(document.readyState != "complete")
    return;

  let dataHolder = document.getElementById('timelineData');
  let data = JSON.parse(dataHolder.innerHTML);
  let offset = Number.parseInt(dataHolder.getAttribute('currentindex'));

  let labels = [];
  let min = 0;
  let max = 0;
  for(let i = 0; i < data.length; i++) {
    labels.push((i - offset).toString());
    if(data[i] < min) {
      min = data[i];
    }
    if(data[i] > max) {
      max = data[i];
    }
  }

  let chart = document.getElementById('timelineChart');
  chart.height = Math.max(200, window.innerWidth / 200 * (Math.abs(min) + max));
  chart.width = window.innerWidth - 100;

  new Chart(chart, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Tagesverlauf',
        data: data,
        fill: false,
        borderColor: '#4f48c4',
        backgroundColor: '#4f48c4',
        tension: 0
      }],
    },
    options: {
      plugins: {
        legend: {
          display: false,
        }
      }
    }
  });
}

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
