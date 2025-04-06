document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");

    form.addEventListener("submit", function () {
        const button = form.querySelector("button");
        button.innerText = "Memproses...";
        button.disabled = true;
    });
});
