function toggleDetails() {
    var details = document.getElementById('details-content');
    var toggle  = document.getElementById('details-flip');
    var btn     = details.previousElementSibling;

    var isHidden = details.classList.contains('details-content--hidden');
    details.classList.toggle('details-content--hidden', !isHidden);
    toggle.classList.toggle('flip', !isHidden);
    if (btn) btn.setAttribute('aria-expanded', String(isHidden));
}
